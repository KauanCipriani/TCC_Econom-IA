"""
Centro de Alertas: histórico dos alertas de compra e venda e seus resultados.

Um alerta é emitido quando o sinal de um ativo muda para compra ou venda.
Depois de cinco pregões o sistema confere se o preço andou na direção do
alerta, o que permite ao usuário acompanhar, com dados reais, a taxa de acerto
de cada modelo e de cada ativo. O histórico pode ser filtrado e exportado.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from econom_ia.config import ATIVOS
from econom_ia.interface import formatacao as fmt
from econom_ia.interface.telas.base import Tela
from econom_ia.servicos import CentralAlertas


class TelaCentroAlertas(Tela):
    chave = "centro_alertas"
    rotulo = "Centro de Alertas"
    icone = ":material/notifications:"
    SECOES_OPCIONAIS = {
        "explicacao": "Como os alertas funcionam",
        "hoje": "Alertas do último pregão",
        "tabela": "Histórico de alertas",
        "por_acao": "Taxa de acerto por ação",
    }

    PERIODOS = ("Desde o fim do treinamento", "Últimos 30 dias", "Últimos 90 dias", "Todo o histórico")
    ROTULOS_TIPO = {"compra": "▲ Compra", "venda": "▼ Venda"}
    ROTULOS_RESULTADO = {"Acerto": "✔ Acerto", "Erro": "✖ Erro", "Aguardando": "… Aguardando"}

    def renderizar(self) -> None:
        self.cabecalho("Centro de Alertas", "Alertas de compra e venda emitidos pelos modelos e o que aconteceu depois")
        if self.aviso_sem_modelo():
            return
        horizonte = self.ctx.horizonte
        if self.mostrar("explicacao"):
            self.ui.renderizar(self.ui.nota(
                f"Um alerta é emitido quando o sinal de uma ação <strong>muda</strong> para compra ou venda. "
                f"{horizonte} pregões depois, o sistema confere o preço: o alerta de compra acerta se a ação subiu e "
                f"o de venda, se caiu. Os modelos foram treinados com dados até "
                f"{fmt.data_curta(self._fim_treinamento())}; alertas anteriores a essa data estão dentro do período "
                "de treino e tendem a parecer melhores do que são."))

        chave_modelo, inicio, tickers, tipos, confianca = self._filtros()
        historico = self.ctx.alertas(chave_modelo, confianca, inicio)
        historico = historico[historico["ticker"].isin(tickers) & historico["tipo"].isin(tipos)]

        self._resumo(historico)
        if self.mostrar("hoje"):
            self._alertas_do_dia(historico)
        if self.mostrar("tabela"):
            self._tabela(historico, chave_modelo)
        if self.mostrar("por_acao"):
            self._acerto_por_ativo(historico)
        self.ui.renderizar(self.ui.aviso_cvm())

    def _fim_treinamento(self) -> pd.Timestamp | None:
        inicio = self.ctx.inicio_prospectivo
        return None if inicio is None else inicio - pd.Timedelta(days=1)

    # ── Filtros ────────────────────────────────────────────────────────────
    def _filtros(self) -> tuple[str, pd.Timestamp | None, list[str], list[str], float]:
        linha1 = st.columns([1.2, 1.4, 1])
        with linha1[0]:
            chave_modelo = self.seletor_modelo()
        with linha1[1]:
            periodo = st.selectbox("Período", self.PERIODOS, key="periodo_alertas")
        with linha1[2]:
            lstm = chave_modelo == "lstm"
            confianca = st.slider("Confiança mínima (%)", 0, 90, 0, step=5, key="confianca_alertas",
                                  disabled=lstm, help="A LSTM não estima probabilidades; para ela o filtro "
                                                      "não se aplica." if lstm else
                                  "Mostra apenas alertas em que o modelo tinha pelo menos essa confiança.")
        linha2 = st.columns([2.2, 1])
        with linha2[0]:
            tickers = st.multiselect("Ações", list(ATIVOS), default=list(ATIVOS), key="ativos_alertas")
        with linha2[1]:
            tipos = st.multiselect("Tipo de alerta", ["compra", "venda"], default=["compra", "venda"],
                                   key="tipos_alertas", format_func=lambda t: self.ROTULOS_TIPO[t])

        ultima = self.ctx.cotacoes.ultima_data
        inicio = {
            "Desde o fim do treinamento": self.ctx.inicio_prospectivo,
            "Últimos 30 dias": ultima - pd.Timedelta(days=30),
            "Últimos 90 dias": ultima - pd.Timedelta(days=90),
            "Todo o histórico": None,
        }[periodo]
        return chave_modelo, inicio, tickers or list(ATIVOS), tipos or ["compra", "venda"], 0 if lstm else confianca

    # ── Seções ─────────────────────────────────────────────────────────────
    def _alertas_do_dia(self, historico: pd.DataFrame) -> None:
        ultima = self.ctx.cotacoes.ultima_data
        hoje = historico[historico["data"] == ultima]
        self.ui.renderizar(self.ui.titulo_secao("Alertas do último pregão", fmt.data_curta(ultima)))
        if hoje.empty:
            self.ui.renderizar(self.ui.nota("Nenhum alerta novo no último pregão: os sinais das ações "
                                            "selecionadas não mudaram para compra ou venda."))
            return
        for _, alerta in hoje.iterrows():
            confianca = ("" if fmt.ausente(alerta["confianca"])
                         else f" com confiança de {fmt.numero(alerta['confianca'], 0)}%")
            acao = "subida" if alerta["tipo"] == "compra" else "queda"
            self.ui.renderizar(self.ui.banner(
                alerta["tipo"], f"Novo alerta · {alerta['ativo']}",
                f"{self.ROTULOS_TIPO[alerta['tipo']].upper()} {alerta['ticker']} a {fmt.moeda(alerta['preco'])}",
                f"O modelo passou a indicar {alerta['tipo']}{confianca}. O resultado (se houve {acao}) será "
                f"conferido em {self.ctx.horizonte} pregões."))

    def _resumo(self, historico: pd.DataFrame) -> None:
        resumo = CentralAlertas.resumir(historico)
        self.ui.renderizar(self.ui.titulo_secao("Resumo do período escolhido",
                                                "quantos alertas houve e quantos acertaram a direção do preço"))
        taxa = fmt.AUSENTE if resumo.taxa_acerto is None else f"{fmt.numero(resumo.taxa_acerto * 100, 1)}%"
        self.ui.renderizar(self.ui.grade([
            self.ui.cartao_valor("Alertas no período", fmt.numero(resumo.total, 0),
                                 detalhe="vezes em que o sinal mudou para compra ou venda"),
            self.ui.cartao_valor("Já avaliados", fmt.numero(resumo.avaliados, 0),
                                 detalhe=f"alertas com {self.ctx.horizonte} pregões completos após a emissão"),
            self.ui.cartao_valor("Taxa de acerto", taxa,
                                 detalhe="o preço andou na direção indicada; um palpite aleatório fica perto de 50%"),
            self.ui.cartao_valor("Retorno médio a favor", fmt.percentual(resumo.retorno_medio_pct),
                                 detalhe="variação média no sentido do alerta, sem custos de operação"),
        ]))

    def _tabela(self, historico: pd.DataFrame, chave_modelo: str) -> None:
        self.ui.renderizar(self.ui.titulo_secao("Histórico de alertas", f"{len(historico)} alertas"))
        if historico.empty:
            st.info("Nenhum alerta com os filtros escolhidos.")
            return
        preco_final = f"Preço após {self.ctx.horizonte} pregões (R$)"
        tabela = pd.DataFrame({
            "Data": historico["data"].dt.strftime("%d/%m/%Y"),
            "Ticker": historico["ticker"],
            "Instituição": historico["ativo"],
            "Alerta": historico["tipo"].map(self.ROTULOS_TIPO),
            "Preço (R$)": historico["preco"].astype(float),
            "Confiança (%)": pd.to_numeric(historico["confianca"], errors="coerce"),
            preco_final: pd.to_numeric(historico["preco_final"], errors="coerce"),
            "Variação (%)": pd.to_numeric(historico["retorno_pct"], errors="coerce"),
            "Resultado": historico["resultado"].map(self.ROTULOS_RESULTADO),
        })
        reais = fmt.formatador(2)
        self.tabela(tabela, {"Preço (R$)": reais, "Confiança (%)": fmt.formatador(0), preco_final: reais,
                             "Variação (%)": fmt.formatador(2, sinal=True)},
                    height=min(420, 38 + 35 * len(tabela)))
        exportar = historico.assign(data=historico["data"].dt.strftime("%d/%m/%Y"))
        st.download_button(
            "Baixar o histórico (CSV)", exportar.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name=f"alertas_{chave_modelo}_{self.ctx.cotacoes.ultima_data:%Y%m%d}.csv", mime="text/csv",
            icon=":material/download:")

    def _acerto_por_ativo(self, historico: pd.DataFrame) -> None:
        resumo = CentralAlertas.resumir(historico)
        if resumo.por_ativo.empty:
            return
        self.ui.renderizar(self.ui.titulo_secao("Taxa de acerto por ação", "alertas já avaliados"))
        por_ativo = resumo.por_ativo
        rotulos = [f"{linha.ticker} ({int(linha.alertas)} alertas)" for linha in por_ativo.itertuples()]
        self.exibir(self.graficos.barras_simples(
            rotulos, list(por_ativo["taxa_acerto"] * 100), "Taxa de acerto (%)", formato="{:.0f}%",
            referencia=50, rotulo_referencia="Acerto esperado ao acaso (50%)"), "grafico_acerto_ativos")
        st.caption("Com poucos alertas avaliados, a taxa de acerto varia muito de um período para outro. Compare "
                   "sempre com a referência de 50%, que é o acerto esperado de um palpite aleatório sobre a "
                   "direção do preço.")
