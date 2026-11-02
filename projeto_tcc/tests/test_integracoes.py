"""Integrações com APIs: interpretação das respostas e política de cache."""

import json
from datetime import datetime

import pandas as pd
import pytest
import requests

from econom_ia.integracoes import (
    ClienteBCB,
    ClienteFocus,
    ClienteFRED,
    ClienteHTTP,
    ClienteNoticias,
    ClientePolymarket,
    ErroIntegracao,
)


class RespostaFalsa:
    def __init__(self, conteudo):
        self.conteudo = conteudo

    def raise_for_status(self):
        pass

    def json(self):
        return self.conteudo

    @property
    def text(self):
        return self.conteudo


class SessaoFalsa:
    """Simula a biblioteca requests; conta as requisições feitas."""

    def __init__(self, conteudo=None, erro: Exception | None = None):
        self.conteudo, self.erro, self.chamadas = conteudo, erro, 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.chamadas += 1
        if self.erro:
            raise self.erro
        return RespostaFalsa(self.conteudo(url, params or {}) if callable(self.conteudo) else self.conteudo)


# ── Cache ──────────────────────────────────────────────────────────────────
def test_politica_de_cache(tmp_path):
    sessao = SessaoFalsa({"valor": 1})
    cliente = ClienteHTTP(pasta_cache=tmp_path, validade_s=3600, sessao=sessao)
    assert cliente.obter_json("https://exemplo/api").origem == "online"
    assert cliente.obter_json("https://exemplo/api").origem == "cache"
    assert sessao.chamadas == 1

    fora_do_ar = ClienteHTTP(pasta_cache=tmp_path, validade_s=0, sessao=SessaoFalsa(erro=requests.ConnectionError()))
    resposta = fora_do_ar.obter_json("https://exemplo/api")
    assert resposta.origem == "cache_antigo" and resposta.conteudo == {"valor": 1}

    offline = ClienteHTTP(pasta_cache=tmp_path, validade_s=0, sessao=SessaoFalsa({"valor": 2}), offline=True)
    assert offline.obter_json("https://exemplo/api").conteudo == {"valor": 1}
    with pytest.raises(ErroIntegracao):
        offline.obter_json("https://exemplo/outra")


# ── Banco Central e FRED ───────────────────────────────────────────────────
def test_bcb_descarta_datas_futuras_e_valores_invalidos():
    conteudo = [{"data": "01/09/2026", "valor": "14.25"}, {"data": "17/09/2026", "valor": "13.75"},
                {"data": "30/12/2026", "valor": "13.75"}, {"data": "18/09/2026", "valor": ""}]
    dados = ClienteBCB.interpretar(conteudo, ate=pd.Timestamp("2026-09-28"))
    assert list(dados["valor"]) == [14.25, 13.75]


@pytest.mark.parametrize("cabecalho", ["observation_date", "DATE"])
def test_fred_aceita_os_dois_formatos_de_cabecalho(cabecalho):
    texto = f"{cabecalho},DGS10\n2026-09-24,5.18\n2026-09-25,.\n2026-09-28,5.17\n"
    dados = ClienteFRED.interpretar(texto)
    assert list(dados["valor"]) == [5.18, 5.17]


# ── Relatório Focus ────────────────────────────────────────────────────────
def test_focus_ordena_as_reunioes_e_usa_a_pesquisa_mais_recente():
    linha = {"Media": 13.6, "DesvioPadrao": 0.1, "Minimo": 13.5, "Maximo": 14.0, "numeroRespondentes": 130}
    conteudo = {"value": [
        {**linha, "Data": "2026-09-25", "Reuniao": "R1/2027", "Mediana": 13.25},
        {**linha, "Data": "2026-09-25", "Reuniao": "R8/2026", "Mediana": 13.5},
        {**linha, "Data": "2026-09-25", "Reuniao": "R7/2026", "Mediana": 13.5},
        {**linha, "Data": "2026-09-18", "Reuniao": "R7/2026", "Mediana": 13.75},
    ]}
    data, reunioes = ClienteFocus.interpretar(conteudo)
    assert data == pd.Timestamp("2026-09-25")
    assert list(reunioes["reuniao"]) == ["R7/2026", "R8/2026", "R1/2027"]
    assert reunioes.iloc[0]["rotulo"] == "7ª reunião de 2026"
    with pytest.raises(ValueError):
        ClienteFocus.interpretar({"value": []})


# ── Polymarket ─────────────────────────────────────────────────────────────
EVENTO = {
    "title": "Bank of Brazil decision in November?", "slug": "bank-of-brazil-decision-in-november-2026",
    "endDate": "2026-11-04T11:59:00Z", "closed": False, "active": True, "volume": "1500.5",
    "markets": [
        {"groupItemTitle": "No Change", "outcomes": '["Yes", "No"]', "outcomePrices": '["0.22", "0.78"]'},
        {"groupItemTitle": "25 bps decrease", "outcomes": '["Yes", "No"]', "outcomePrices": '["0.7", "0.3"]'},
        {"groupItemTitle": "50+ bps decrease", "outcomes": '["Yes", "No"]', "outcomePrices": '["0.08", "0.92"]',
         "active": False},
    ],
}


def test_polymarket_converte_e_traduz_os_resultados():
    mercado = ClientePolymarket.converter(EVENTO)
    assert mercado.opcoes == [("Corte de 0,25 p.p.", 0.7), ("Manutenção da Selic", 0.22)]
    assert mercado.mais_provavel == ("Corte de 0,25 p.p.", pytest.approx(0.7 / 0.92))  # normalizada
    assert ClientePolymarket.traduzir("50+ bps increase") == "Alta de 0,50 p.p. ou mais"


def test_polymarket_ignora_eventos_arquivados_ou_encerrados():
    hoje = pd.Timestamp("2026-09-28")
    assert ClientePolymarket.eh_valido(EVENTO, hoje)
    assert not ClientePolymarket.eh_valido({**EVENTO, "archived": True}, hoje)
    assert not ClientePolymarket.eh_valido({**EVENTO, "closed": True}, hoje)
    assert not ClientePolymarket.eh_valido({**EVENTO, "title": "Brazil Presidential Election"}, hoje)


def test_polymarket_encontra_o_evento_da_proxima_reuniao(tmp_path):
    def responder(url, params):
        if params.get("slug") == EVENTO["slug"]:
            return [EVENTO]
        return {"events": []} if "public-search" in url else []
    cliente = ClientePolymarket(pasta_cache=tmp_path, sessao=SessaoFalsa(responder))
    mercado = cliente.proxima_decisao_copom(hoje=pd.Timestamp("2026-09-28"))
    assert mercado is not None and mercado.url.endswith(EVENTO["slug"])


def test_polymarket_interrompe_a_busca_na_primeira_falha_de_conexao(tmp_path):
    """Com o acesso bloqueado, a busca não pode travar a interface tentando todos os endereços."""
    sessao = SessaoFalsa(erro=requests.ConnectionError())
    with pytest.raises(ErroIntegracao):
        ClientePolymarket(pasta_cache=tmp_path, sessao=sessao).proxima_decisao_copom(hoje=pd.Timestamp("2026-09-28"))
    assert sessao.chamadas == 1


# ── Notícias ───────────────────────────────────────────────────────────────
RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Notícia Itaú anuncia recompra de ações - Valor</title><link>https://exemplo/1</link>
<pubDate>Mon, 28 Sep 2026 17:00:00 GMT</pubDate><source url="https://valor.globo.com">Valor</source></item>
<item><title>ITUB4: cotação hoje, dividendos e indicadores - Site</title><link>https://exemplo/2</link>
<pubDate>Mon, 28 Sep 2026 12:00:00 GMT</pubDate><source url="https://site">Site</source></item>
</channel></rss>"""


def test_noticias_interpreta_o_rss_e_filtra_paginas_de_cotacao(tmp_path):
    noticias = ClienteNoticias.interpretar_rss(RSS, "ITUB4")
    assert noticias[0].titulo == "Itaú anuncia recompra de ações"
    assert noticias[0].fonte == "Valor"
    assert noticias[0].publicada_em == datetime(2026, 9, 28, 14, 0)  # 17h GMT = 14h em Brasília
    assert not ClienteNoticias.eh_noticia(noticias[1].titulo)

    resultado = ClienteNoticias(pasta_cache=tmp_path, sessao=SessaoFalsa(RSS)).manchetes()
    assert [n.titulo for n in resultado.noticias] == ["Itaú anuncia recompra de ações"]
    assert resultado.origem == "online"


def test_cache_em_disco_e_json_legivel(tmp_path):
    ClienteBCB(pasta_cache=tmp_path, sessao=SessaoFalsa([{"data": "01/09/2026", "valor": "14.25"}])).serie("selic_meta")
    arquivos = list(tmp_path.glob("bcb_*.json"))
    assert len(arquivos) == 1
    assert json.loads(arquivos[0].read_text(encoding="utf-8"))["conteudo"][0]["valor"] == "14.25"


def test_copia_local_continua_valendo_em_outro_dia(tmp_path):
    """As datas de início e fim mudam todo dia e não entram no nome da cópia local."""
    bcb = ClienteBCB(pasta_cache=tmp_path)
    url = bcb.URL.format(codigo=432)
    ontem = bcb._chave(url, {"formato": "json", "dataInicial": "27/09/2021", "dataFinal": "27/09/2026"})
    hoje = bcb._chave(url, {"formato": "json", "dataInicial": "28/09/2021", "dataFinal": "28/09/2026"})
    assert ontem == hoje
    assert bcb._chave(url, {"formato": "json"}) != bcb._chave(bcb.URL.format(codigo=13522), {"formato": "json"})


def test_fred_usa_a_api_oficial_quando_ha_chave(tmp_path):
    recebidos = {}

    def responder(url, params):
        recebidos.update(url=url, **params)
        return {"observations": [{"date": "2026-10-02", "value": "4.10"}, {"date": "2026-10-01", "value": "."}]}

    cliente = ClienteFRED(pasta_cache=tmp_path, sessao=SessaoFalsa(responder), chave_api="abc123")
    serie = cliente.serie("fed_funds")
    assert recebidos["url"] == ClienteFRED.URL_API and recebidos["api_key"] == "abc123"
    assert serie.ultimo_valor == pytest.approx(4.10) and len(serie.dados) == 1  # "." é descartado
    assert not any("abc123" in arquivo.name for arquivo in tmp_path.iterdir())


def test_falha_de_rede_e_tentada_de_novo(tmp_path, monkeypatch):
    class SessaoInstavel(SessaoFalsa):
        def get(self, url, params=None, headers=None, timeout=None):
            self.chamadas += 1
            if self.chamadas == 1:
                raise requests.Timeout("lenta")
            return RespostaFalsa([{"data": "01/09/2026", "valor": "14.25"}])

    monkeypatch.setattr(ClienteHTTP, "ESPERA_ENTRE_TENTATIVAS_S", 0)
    sessao = SessaoInstavel()
    serie = ClienteBCB(pasta_cache=tmp_path, sessao=sessao).serie("selic_meta")
    assert sessao.chamadas == 2 and serie.ultimo_valor == pytest.approx(14.25)


def test_chave_de_acesso_nao_aparece_no_log():
    assert ClienteHTTP._sem_segredos("erro em ...?api_key=segredo&file_type=json") == (
        "erro em ...?api_key=***&file_type=json")


def test_polymarket_normaliza_as_probabilidades_e_marca_pouco_volume():
    from econom_ia.integracoes import MercadoPrevisao

    mercado = MercadoPrevisao("Copom", "https://polymarket.com/event/x", None, 1400.0,
                              [("Manutenção da Selic", 0.40), ("Corte de 0,50 p.p. ou mais", 0.30),
                               ("Alta de 0,25 p.p.", 0.18), ("Corte de 0,25 p.p.", 0.18),
                               ("Alta de 0,50 p.p. ou mais", 0.10)])
    assert mercado.soma_bruta == pytest.approx(1.16)
    assert sum(p for _, p in mercado.opcoes_normalizadas) == pytest.approx(1.0)
    assert mercado.mais_provavel == ("Manutenção da Selic", pytest.approx(0.40 / 1.16))
    assert mercado.pouco_volume
