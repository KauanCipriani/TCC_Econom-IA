"""Serviços da interface: alertas, calendário de pregões e faixa da previsão."""

import numpy as np
import pandas as pd
import pytest

from econom_ia.servicos import CentralAlertas, ServicoPrevisao
from econom_ia.servicos.calendario import proximos_pregoes


def _sinais():
    datas = pd.bdate_range("2026-08-03", periods=12)
    sinais = ["neutro", "compra", "compra", "neutro", "neutro", "venda", "venda", "compra",
              "neutro", "neutro", "neutro", "venda"]
    precos = [10, 10, 10.2, 10.4, 10.5, 10.6, 10.4, 10.1, 10.0, 10.3, 10.2, 10.1]
    confianca = [40, 45, 45, 40, 40, 70, 70, 30, 40, 40, 40, 60]
    return pd.DataFrame({"data": datas, "ticker": "ITUB4", "close": precos, "sinal": sinais,
                         "confianca": confianca})


def test_alerta_apenas_quando_o_sinal_muda():
    historico = CentralAlertas(horizonte=3).historico(_sinais())
    assert list(historico["tipo"]) == ["venda", "compra", "venda", "compra"]  # mais recentes primeiro
    compra_inicial = historico.iloc[-1]
    assert compra_inicial["preco_final"] == pytest.approx(10.5)  # 3 pregões depois do alerta a R$ 10,00
    assert compra_inicial["resultado"] == "Acerto"
    assert historico.iloc[0]["resultado"] == "Aguardando"


def test_filtro_de_confianca_e_resumo():
    central = CentralAlertas(horizonte=3, confianca_minima=50)
    historico = central.historico(_sinais())
    assert set(historico["confianca"]) == {70, 60}
    resumo = CentralAlertas.resumir(historico)
    assert resumo.total == 2 and resumo.aguardando == 1 and resumo.avaliados == 1
    assert resumo.taxa_acerto == pytest.approx(1.0)  # venda a 10,6 e preço 3 pregões depois a 10,0


def test_calendario_pula_fins_de_semana_e_feriados():
    datas = proximos_pregoes(pd.Timestamp("2026-10-09"), 2)  # sexta-feira antes do feriado de 12/10
    assert list(datas) == [pd.Timestamp("2026-10-13"), pd.Timestamp("2026-10-14")]


def test_faixa_da_previsao_acompanha_a_volatilidade():
    servico = ServicoPrevisao.__new__(ServicoPrevisao)
    servico.faixas = {"1": {"q10": -0.01, "q90": 0.01, "z10": -1.2, "z90": 1.3},
                      "2": {"q10": -0.02, "q90": 0.02, "z10": -1.3, "z90": 1.4}}
    dados = pd.DataFrame({"volatilidade_20": [0.02]})
    inferior, superior = servico._limites_em_retorno(dados, np.array([1, 2]))
    assert inferior == pytest.approx([-1.2 * 0.02, -1.3 * 0.02 * np.sqrt(2)])
    assert superior == pytest.approx([1.3 * 0.02, 1.4 * 0.02 * np.sqrt(2)])
