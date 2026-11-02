"""
Tela Previsões & Macro: contexto macroeconômico do Brasil e dos EUA.

Integra fontes públicas: a API de séries temporais do Banco Central (Selic,
IPCA e dólar PTAX), o Relatório Focus do Banco Central (expectativas do
mercado para a Selic em cada reunião do Copom), o FRED do Federal Reserve de
St. Louis (Fed Funds e juros dos títulos do Tesouro americano) e o mercado de
previsão Polymarket (probabilidades para a próxima decisão do Copom). Os dados
são contextuais: não entram nos modelos, que usam apenas indicadores técnicos.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from econom_ia.integracoes import ExpectativasSelic, MercadoPrevisao, SerieTemporal
from econom_ia.interface import dados_app
from econom_ia.interface import formatacao as fmt
from econom_ia.interface.telas.base import Tela
from econom_ia.interface.telas.visao_geral import TelaVisaoGeral

NOMES_SERIES = {
    "selic_meta": ("Meta Selic", "Banco Central (SGS 432)"),
    "ipca_12m": ("IPCA em 12 meses", "Banco Central (SGS 13522)"),
    "dolar_ptax": ("Dólar PTAX (venda)", "Banco Central (SGS 1)"),
    "fed_funds": ("Fed Funds (taxa efetiva)", "FRED (DFF)"),
    "treasury_3m": ("Tesouro americano 3 meses", "FRED (DGS3MO)"),
    "treasury_2a": ("Tesouro americano 2 anos", "FRED (DGS2)"),
    "treasury_10a": ("Tesouro americano 10 anos", "FRED (DGS10)"),
}


class TelaPrevisoesMacro(Tela):
    chave = "previsoes_macro"
    rotulo = "Previsões & Macro"
    icone = ":material/public:"
    SECOES_OPCIONAIS = {
        "brasil": "Indicadores do Brasil",
        "eua": "Indicadores dos Estados Unidos",
        "graficos": "Gráficos de juros e inflação",
        "focus": "Expectativas do Relatório Focus",
        "copom": "Polymarket (próxima decisão do Copom)",
    }

    def renderizar(self) -> None:
        self.cabecalho("Previsões & Macro",
                       "Juros, inflação e câmbio no Brasil e nos Estados Unidos, e o que o mercado espera do Copom")

        with st.spinner("Consultando Banco Central, FRED e Polymarket..."):
            internet = self.ctx.usar_internet
            series = {chave: dados_app.serie_bcb(chave, internet) for chave in ("selic_meta", "ipca_12m", "dolar_ptax")}
            series |= {chave: dados_app.serie_fred(chave, internet)
                       for chave in ("fed_funds", "treasury_3m", "treasury_2a", "treasury_10a")}
            focus = dados_app.expectativas_focus(internet)
            mercado, polymarket_respondeu = dados_app.decisao_copom(internet)

        self._leitura(series, focus)
        if self.mostrar("brasil"):
            self._brasil(series)
        if self.mostrar("eua"):
            self._estados_unidos(series)
        if self.mostrar("graficos"):
            self._graficos(series)
        if self.mostrar("focus"):
            self._expectativas(focus, series["selic_meta"])
        if self.mostrar("copom"):
            self._copom(mercado, polymarket_respondeu)
        self._fontes(series, focus, mercado, polymarket_respondeu)
        self.ui.renderizar(self.ui.aviso_cvm())

    # ── Cartões ────────────────────────────────────────────────────────────
    @staticmethod
    def _disponivel(serie: SerieTemporal | None) -> bool:
        return serie is not None and serie.ultimo_valor is not None

    def _cartao_taxa(self, rotulo: str, serie: SerieTemporal | None, detalhe: str = "") -> str:
        if not self._disponivel(serie):
            return self.ui.cartao_valor(rotulo, fmt.AUSENTE, detalhe="Fonte indisponível no momento")
        return self.ui.cartao_valor(rotulo, f"{fmt.numero(serie.ultimo_valor)}%", "a.a.", detalhe=detalhe,
                                    fonte=f"{fmt.data_curta(serie.ultima_data)} · "
                                          f"{self.ui.origem(serie.origem, serie.obtido_em)}")

    def _variacao_12m(self, serie: SerieTemporal | None) -> str:
        if not self._disponivel(serie):
            return ""
        anterior = serie.valor_em(365)
        if anterior is None:
            return ""
        return f"{fmt.pontos_percentuais(serie.ultimo_valor - anterior)} em 12 meses"

    def _brasil(self, series: dict) -> None:
        self.ui.renderizar(self.ui.titulo_secao("Brasil"))
        selic, ipca, ptax = series["selic_meta"], series["ipca_12m"], series["dolar_ptax"]
        indisponivel = "Fonte indisponível no momento"
        cartoes = []

        detalhe = ""
        if self._disponivel(selic):
            detalhe = f"em vigor desde {fmt.data_curta(TelaVisaoGeral.ultima_mudanca(selic))}"
        cartoes.append(self._cartao_taxa("Selic (meta do Copom)", selic, detalhe))

        if self._disponivel(ipca):
            cartoes.append(self.ui.cartao_valor(
                "IPCA em 12 meses", f"{fmt.numero(ipca.ultimo_valor)}%",
                detalhe=f"até {fmt.mes_ano(ipca.ultima_data)} · {self._variacao_12m(ipca)}",
                fonte=self.ui.origem(ipca.origem, ipca.obtido_em)))
        else:
            cartoes.append(self.ui.cartao_valor("IPCA em 12 meses", fmt.AUSENTE, detalhe=indisponivel))

        juro_real = self.juro_real(selic, ipca)
        cartoes.append(self.ui.cartao_valor(
            "Juro real", fmt.AUSENTE if juro_real is None else f"{fmt.numero(juro_real)}%", "a.a.",
            detalhe="Selic descontada a inflação de 12 meses", fonte="(1 + Selic) / (1 + IPCA) − 1"))

        if self._disponivel(ptax):
            anterior = ptax.valor_em(30)
            variacao = None if not anterior else (ptax.ultimo_valor / anterior - 1) * 100
            cartoes.append(self.ui.cartao_valor(
                "Dólar PTAX", fmt.moeda(ptax.ultimo_valor, 4), detalhe=f"{self.ui.variacao(variacao)} em 30 dias",
                fonte=f"{fmt.data_curta(ptax.ultima_data)} · {self.ui.origem(ptax.origem, ptax.obtido_em)}"))
        else:
            cartoes.append(self.ui.cartao_valor("Dólar PTAX", fmt.AUSENTE, detalhe=indisponivel))
        self.ui.renderizar(self.ui.grade(cartoes))

    def _estados_unidos(self, series: dict) -> None:
        self.ui.renderizar(self.ui.titulo_secao("Estados Unidos"))
        cartoes = (
            ("Fed Funds (efetiva)", "fed_funds"),
            ("Tesouro 3 meses", "treasury_3m"),
            ("Tesouro 2 anos", "treasury_2a"),
            ("Tesouro 10 anos", "treasury_10a"),
        )
        self.ui.renderizar(self.ui.grade([
            self._cartao_taxa(rotulo, series[chave], self._variacao_12m(series[chave])) for rotulo, chave in cartoes]))

    # ── Leitura do cenário ─────────────────────────────────────────────────
    @staticmethod
    def juro_real(selic: SerieTemporal | None, ipca: SerieTemporal | None) -> float | None:
        if selic is None or ipca is None or selic.ultimo_valor is None or ipca.ultimo_valor is None:
            return None
        return ((1 + selic.ultimo_valor / 100) / (1 + ipca.ultimo_valor / 100) - 1) * 100

    def _leitura(self, series: dict, focus: ExpectativasSelic | None) -> None:
        """Resumo automático, em texto corrido, dos números exibidos."""
        selic, ipca, fed = series["selic_meta"], series["ipca_12m"], series["fed_funds"]
        curto, longo = series["treasury_3m"], series["treasury_10a"]
        frases = []
        juro_real = self.juro_real(selic, ipca)
        if juro_real is not None:
            frases.append(f"A Selic está em {fmt.numero(selic.ultimo_valor)}% ao ano e a inflação acumulada em 12 "
                          f"meses é de {fmt.numero(ipca.ultimo_valor)}%, o que resulta em juro real de "
                          f"{fmt.numero(juro_real, 1)}% ao ano.")
        if focus is not None and focus.proxima is not None and self._disponivel(selic):
            proxima = focus.proxima
            diferenca = proxima["mediana"] - selic.ultimo_valor
            movimento = ("manutenção da taxa" if abs(diferenca) < 0.01 else
                         f"{'corte' if diferenca < 0 else 'alta'} de {fmt.numero(abs(diferenca))} p.p.")
            frases.append(f"Segundo o Relatório Focus de {fmt.data_curta(focus.data_referencia)}, o mercado espera "
                          f"{movimento} na próxima reunião do Copom (mediana de {fmt.numero(proxima['mediana'])}%).")
        if self._disponivel(selic) and self._disponivel(fed):
            diferencial = selic.ultimo_valor - fed.ultimo_valor
            frases.append(f"Nos Estados Unidos a taxa básica efetiva está em {fmt.numero(fed.ultimo_valor)}%, "
                          f"um diferencial de {fmt.numero(diferencial, 1)} p.p. em relação à Selic.")
        if self._disponivel(curto) and self._disponivel(longo):
            inclinacao = longo.ultimo_valor - curto.ultimo_valor
            formato = "invertida (juros longos abaixo dos curtos)" if inclinacao < 0 else "positivamente inclinada"
            frases.append(f"A curva de juros americana está {formato}: {fmt.pontos_percentuais(inclinacao)} entre "
                          "os títulos de 10 anos e de 3 meses.")
        if frases:
            frases.append("Juros reais elevados tornam a renda fixa mais atrativa em relação às ações e encarecem "
                          "o crédito; o diferencial de juros entre os países influencia o fluxo de capital "
                          "estrangeiro e o câmbio.")
            self.ui.renderizar(self.ui.titulo_secao("Resumo do cenário",
                                                    "o que os números abaixo dizem, em poucas linhas"))
            itens = "".join(f"<li>{frase}</li>" for frase in frases)
            self.ui.renderizar(self.ui.cartao_html(f'<ul class="texto-corrido lista-resumo">{itens}</ul>'))
        self.ui.renderizar(self.ui.nota(
            "Estes indicadores dão contexto econômico às decisões. Eles <strong>não</strong> entram nos modelos de "
            "classificação, que usam apenas os indicadores técnicos calculados a partir das cotações das ações."))

    # ── Gráficos ───────────────────────────────────────────────────────────
    def _graficos(self, series: dict) -> None:
        esquerda, direita = st.columns(2, gap="large")
        with esquerda:
            self.ui.renderizar(self.ui.titulo_secao("Selic e inflação", "últimos 5 anos, % ao ano"))
            brasil = {NOMES_SERIES[c][0]: series[c].dados for c in ("selic_meta", "ipca_12m")
                      if self._disponivel(series[c])}
            if brasil:
                self.exibir(self.graficos.series_temporais(brasil, "% ao ano", degraus=True), "grafico_selic")
            else:
                st.info("Séries do Banco Central indisponíveis no momento.")
        with direita:
            self.ui.renderizar(self.ui.titulo_secao("Juros nos Estados Unidos", "últimos 5 anos, % ao ano"))
            eua = {NOMES_SERIES[c][0]: series[c].dados
                   for c in ("fed_funds", "treasury_3m", "treasury_2a", "treasury_10a")
                   if self._disponivel(series[c])}
            if eua:
                self.exibir(self.graficos.series_temporais(eua, "% ao ano"), "grafico_eua")
            else:
                st.info("Séries do FRED indisponíveis no momento.")

    # ── Expectativas do Relatório Focus ────────────────────────────────────
    def _expectativas(self, focus: ExpectativasSelic | None, selic: SerieTemporal | None) -> None:
        self.ui.renderizar(self.ui.titulo_secao("O que o mercado espera para a Selic",
                                                "Relatório Focus do Banco Central"))
        if focus is None or focus.proxima is None:
            self.ui.renderizar(self.ui.nota("As expectativas do Relatório Focus estão indisponíveis no momento."))
            return
        atual = selic.ultimo_valor if self._disponivel(selic) else None
        fonte = (f"Focus de {fmt.data_curta(focus.data_referencia)} · "
                 f"{self.ui.origem(focus.origem, focus.obtido_em)}")
        cartoes = [("Próxima reunião do Copom", focus.proxima)]
        ano = focus.data_referencia.year
        for rotulo, reuniao in ((f"Selic ao fim de {ano}", focus.fim_de_ano(ano)),
                                (f"Selic ao fim de {ano + 1}", focus.fim_de_ano(ano + 1))):
            if reuniao is not None:
                cartoes.append((rotulo, reuniao))
        colunas = st.columns([1, 2.4], gap="large")
        with colunas[0]:
            for rotulo, reuniao in cartoes:
                variacao = "" if atual is None else (
                    f" · {fmt.pontos_percentuais(reuniao['mediana'] - atual)} sobre a Selic atual")
                self.ui.renderizar(self.ui.cartao_valor(
                    rotulo, f"{fmt.numero(reuniao['mediana'])}%", "a.a.",
                    detalhe=f"mediana para a {reuniao['rotulo']}{variacao}",
                    fonte=f"{reuniao['respondentes']} instituições · {fonte}"))
        with colunas[1]:
            self.exibir(self.graficos.trajetoria_selic(focus.reunioes, atual), "grafico_focus")
            st.caption("Cada ponto é a mediana das projeções das instituições consultadas pelo Banco Central para "
                       "a meta Selic ao fim de uma reunião do Copom (R7/2026 é a sétima reunião de 2026). A faixa "
                       "sombreada vai da menor à maior projeção informada.")

    # ── Mercado de previsão ────────────────────────────────────────────────
    def _copom(self, mercado: MercadoPrevisao | None, respondeu: bool) -> None:
        self.ui.renderizar(self.ui.titulo_secao("Próxima decisão do Copom no mercado de previsão", "Polymarket"))
        if not respondeu:
            self.ui.renderizar(self.ui.nota(
                "Não foi possível consultar o Polymarket. Desde abril de 2026 o acesso à plataforma é bloqueado no "
                "Brasil por determinação da Secretaria de Prêmios e Apostas do Ministério da Fazenda, por isso a "
                "consulta costuma falhar em redes brasileiras. As expectativas do Relatório Focus, acima, cumprem o "
                "mesmo papel com uma fonte oficial."))
            return
        if mercado is None or not mercado.opcoes:
            self.ui.renderizar(self.ui.nota(
                "Não há mercado aberto para a próxima reunião do Copom no Polymarket neste momento. Os mercados "
                "costumam abrir algumas semanas antes de cada reunião."))
            return
        grafico, resumo = st.columns([1.6, 1], gap="large")
        with grafico:
            self.exibir(self.graficos.probabilidades(mercado.opcoes), "grafico_copom")
        with resumo:
            rotulo, probabilidade = mercado.mais_provavel
            volume = "" if mercado.volume is None else f"Volume negociado: US$ {fmt.numero_compacto(mercado.volume)}"
            self.ui.renderizar(self.ui.cartao_valor(
                "Resultado mais provável", self.ui.texto_seguro(rotulo),
                detalhe=f"probabilidade de {fmt.numero(probabilidade * 100, 0)}% · encerra em "
                        f"{fmt.data_curta(mercado.data_encerramento)}",
                fonte=f"{volume} · {self.ui.origem(mercado.origem, mercado.obtido_em)}"))
            st.link_button("Abrir o mercado no Polymarket", mercado.url, icon=":material/open_in_new:",
                           width="stretch")
        st.caption("No Polymarket, cada resultado é negociado como um contrato que paga US$ 1 se acontecer. O preço "
                   "do contrato, entre 0 e 1, funciona como a probabilidade coletiva estimada pelos participantes. "
                   "O sistema apenas lê essas cotações públicas; não há apostas nem cadastro na plataforma.")

    # ── Fontes ─────────────────────────────────────────────────────────────
    def _fontes(self, series: dict, focus: ExpectativasSelic | None, mercado: MercadoPrevisao | None,
                polymarket_respondeu: bool) -> None:
        linhas = []
        for chave, serie in series.items():
            nome, fonte = NOMES_SERIES[chave]
            linhas.append({
                "Indicador": nome,
                "Fonte": fonte,
                "Última observação": fmt.data_curta(serie.ultima_data) if self._disponivel(serie) else fmt.AUSENTE,
                "Situação": self.ui.origem(serie.origem, serie.obtido_em) if serie is not None else "indisponível",
            })
        linhas.append({
            "Indicador": "Expectativas para a Selic",
            "Fonte": "Banco Central (Relatório Focus, API Olinda)",
            "Última observação": fmt.data_curta(focus.data_referencia) if focus else fmt.AUSENTE,
            "Situação": self.ui.origem(focus.origem, focus.obtido_em) if focus else "indisponível",
        })
        situacao_polymarket = ("inacessível" if not polymarket_respondeu else
                               self.ui.origem(mercado.origem, mercado.obtido_em) if mercado else "sem mercado aberto")
        linhas.append({
            "Indicador": "Decisão do Copom (probabilidades)",
            "Fonte": "Polymarket (Gamma API)",
            "Última observação": fmt.data_hora(mercado.obtido_em) if mercado else fmt.AUSENTE,
            "Situação": situacao_polymarket,
        })
        with st.expander("Fontes e atualização dos dados"):
            self.tabela(pd.DataFrame(linhas))
            st.caption("As consultas são guardadas por até 1 hora (15 minutos no Polymarket). Sem conexão, o sistema "
                       "exibe a última cópia salva e informa a data em que ela foi obtida.")
