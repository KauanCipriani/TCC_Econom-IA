"""
Painel Iniciante: a leitura do sinal em linguagem simples.

Responde à pergunta "é um bom momento para investir nesta ação?" com o sinal
do modelo principal, explica o que ele significa na prática e mostra, com
dados reais, quantas vezes os alertas recentes acertaram a direção do preço.
Os termos técnicos ficam para o Painel Avançado.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from econom_ia.config import ATIVOS
from econom_ia.interface import formatacao as fmt
from econom_ia.interface.telas.base import Tela


class TelaPainelIniciante(Tela):
    chave = "painel_iniciante"
    rotulo = "Painel Iniciante"
    icone = ":material/school:"
    SECOES_OPCIONAIS = {
        "significado": "O que isso significa para mim",
        "numeros": "Os números de hoje",
        "grafico": "Gráfico do preço",
        "historico": "Alertas das últimas semanas",
        "avancado": "Convite para o Painel Avançado",
    }

    DIAS_ESTIMATIVA = 5

    # Resposta curta e explicação para cada sinal.
    RESPOSTAS = {
        "compra": (
            "Os indicadores apontam para possível alta",
            "Os sinais técnicos de <strong>{nome}</strong> se parecem com os de momentos do passado que "
            "antecederam altas do preço.",
        ),
        "venda": (
            "Os indicadores pedem cautela",
            "Os sinais técnicos de <strong>{nome}</strong> se parecem com os de momentos do passado que "
            "antecederam quedas do preço.",
        ),
        "neutro": (
            "Sem sinal claro no momento",
            "Os indicadores de <strong>{nome}</strong> estão equilibrados: não há sinal forte de alta nem de "
            "baixa para os próximos dias.",
        ),
    }

    SIGNIFICADOS = {
        "compra": (
            "**Em linguagem simples:** o sistema compara os indicadores de hoje com o histórico da ação e "
            "encontrou uma combinação que, no passado, veio antes de altas. Isso **não garante** que a ação vai "
            "subir.\n\n"
            "**Próximos passos:** se você já pensava em investir nessa ação, este é um bom momento para estudar "
            "a empresa, ler as notícias recentes e, se tiver, conversar com um assessor de investimentos."
        ),
        "venda": (
            "**Em linguagem simples:** o sistema encontrou nos indicadores uma combinação que, no passado, veio "
            "antes de quedas. Isso **não significa** que a ação vai cair com certeza.\n\n"
            "**Próximos passos:** se você já tem essa ação, vale revisar a sua estratégia; se pensava em comprar, "
            "talvez seja melhor acompanhar por mais alguns dias antes de decidir."
        ),
        "neutro": (
            "**Em linguagem simples:** os indicadores não apontam uma direção. Isso costuma acontecer quando o "
            "preço oscila dentro de uma faixa estreita ou quando o mercado está mudando de tendência.\n\n"
            "**Próximos passos:** aguardar costuma ser razoável quando não há direção clara. Os sinais são "
            "recalculados a cada pregão e podem mudar nos próximos dias."
        ),
    }

    def renderizar(self) -> None:
        self.cabecalho("Painel Iniciante", "Uma leitura simples, sem termos técnicos, do momento de cada banco")
        if self.aviso_sem_modelo():
            return
        coluna_ativo, _ = st.columns([1, 2])
        with coluna_ativo:
            ticker = self.seletor_ativo("Qual ação você quer analisar?")
        dados = self.ctx.ativo(ticker)
        if dados.empty:
            st.warning(f"Não há cotações disponíveis para {ticker}.")
            return

        ultimo = dados.iloc[-1]
        sinal = ultimo["sinal"] if ultimo["sinal"] in self.RESPOSTAS else "neutro"
        previsao = self.ctx.servico_previsao.prever(dados, self.DIAS_ESTIMATIVA)
        alertas = self.ctx.alertas()
        alertas = alertas[alertas["ticker"] == ticker]

        self._banner(ticker, ultimo, sinal, alertas)
        if self.mostrar("significado"):
            self._significado(sinal)
        if self.mostrar("numeros"):
            self.ui.renderizar(self.ui.titulo_secao("Os números de hoje"))
            self._cartoes(dados, previsao)
        if self.mostrar("grafico"):
            self._grafico(ticker, dados, previsao)
        if self.mostrar("historico"):
            self._historico(alertas)
        self.ui.renderizar(self.ui.aviso_cvm())
        if self.mostrar("avancado"):
            self._convite_avancado()

    # ── Seções ─────────────────────────────────────────────────────────────
    def _significado(self, sinal: str) -> None:
        self.ui.renderizar(self.ui.titulo_secao("O que isso significa para mim?"))
        with st.container(border=True):
            st.markdown(self.SIGNIFICADOS[sinal])
            st.markdown("**Lembre-se:** nos testes com dados que os modelos nunca tinham visto, eles acertaram "
                        "pouco mais do que um palpite aleatório. Use o sinal como mais uma informação, nunca como "
                        "a única base para decidir.")

    def _grafico(self, ticker: str, dados: pd.DataFrame, previsao: pd.DataFrame) -> None:
        self.ui.renderizar(self.ui.titulo_secao("Como a ação se comportou?"))
        pregoes = self.seletor_periodo()
        self.exibir(self.graficos.preco_simples(dados, previsao, ticker, pregoes), "grafico_iniciante")
        if not previsao.empty:
            st.caption(f"A linha tracejada é a estimativa para os próximos {self.DIAS_ESTIMATIVA} pregões. A área "
                       "sombreada é a faixa provável: nos testes, o preço terminou dentro de faixas assim em cerca "
                       "de 8 a cada 10 vezes.")

    def _convite_avancado(self) -> None:
        with st.expander("Quer ver a análise técnica completa?"):
            st.markdown("O **Painel Avançado** mostra os indicadores técnicos (RSI, MACD, Bandas de Bollinger e "
                        "médias móveis), a opinião de cada um dos quatro modelos, os cenários de preço e o que "
                        "mais pesou no sinal. É indicado para quem já conhece análise técnica.")
            self.botao_navegacao("Abrir o Painel Avançado", "painel_avancado", ":material/query_stats:")

    def _banner(self, ticker: str, ultimo: pd.Series, sinal: str, alertas: pd.DataFrame) -> None:
        nome = ATIVOS[ticker].nome
        resposta, explicacao = self.RESPOSTAS[sinal]
        partes = [explicacao.format(nome=nome)]
        if not fmt.ausente(ultimo["confianca"]):
            # A confiança indica o quanto o modelo prefere este cenário aos outros dois; ela não é
            # calibrada e não deve ser lida como chance de acerto (que vem logo em seguida).
            partes.append(f"A confiança do modelo nesse cenário é de "
                          f"<strong>{fmt.numero(ultimo['confianca'], 0)}%</strong>, o que não equivale à chance "
                          "de acerto.")
        partes.append(self._frase_acertos(ticker, alertas))
        icone = {"compra": "▲", "venda": "▼", "neutro": "●"}[sinal]
        self.ui.renderizar(self.ui.banner(sinal, f"É um bom momento para investir em {ticker}?",
                                          f"{icone} {resposta}", " ".join(partes)))

    def _frase_acertos(self, ticker: str, alertas: pd.DataFrame) -> str:
        """Taxa de acerto dos alertas emitidos depois do período de treinamento."""
        inicio = self.ctx.inicio_prospectivo
        if inicio is not None:
            alertas = alertas[alertas["data"] >= inicio]
        avaliados = alertas[alertas["resultado"] != "Aguardando"]
        desde = f"desde {fmt.mes_ano(inicio)}" if inicio is not None else "no histórico"
        if avaliados.empty:
            return f"Ainda não há alertas avaliados para {ticker} {desde}."
        acertos = int((avaliados["resultado"] == "Acerto").sum())
        return (f"Os alertas para {ticker} emitidos {desde}, período que o modelo não viu no treinamento, "
                f"acertaram a direção do preço em <strong>{acertos} de {len(avaliados)}</strong> vezes "
                f"({fmt.numero(acertos / len(avaliados) * 100, 0)}%).")

    def _cartoes(self, dados: pd.DataFrame, previsao: pd.DataFrame) -> None:
        ultimo = dados.iloc[-1]
        anterior = dados.iloc[-2] if len(dados) > 1 else ultimo
        mes = dados.iloc[-22] if len(dados) > 22 else dados.iloc[0]
        variacao_dia = (ultimo["close"] / anterior["close"] - 1) * 100
        variacao_mes = (ultimo["close"] / mes["close"] - 1) * 100
        diferenca = ultimo["close"] - mes["close"]
        verbo = "subiu" if diferenca > 0 else "caiu" if diferenca < 0 else "ficou estável,"
        titulo_estimativa = f"Estimativa para {self.DIAS_ESTIMATIVA} pregões"

        cartoes = [
            self.ui.cartao_valor(
                "Preço no último pregão", fmt.moeda(ultimo["close"]),
                detalhe=f"{self.ui.variacao(variacao_dia)} desde o pregão anterior",
                fonte=f"Fechamento de {fmt.data_curta(ultimo['data'])}"),
            self.ui.cartao_valor(
                "No último mês", self.ui.variacao(variacao_mes),
                detalhe=f"A ação {verbo} {fmt.moeda(abs(diferenca))} em 21 pregões"),
        ]
        if previsao.empty:
            cartoes.append(self.ui.cartao_valor(
                titulo_estimativa, fmt.AUSENTE, detalhe=self.ctx.servico_previsao.erro or "Estimativa indisponível"))
        else:
            final = previsao.iloc[-1]
            cartoes.append(self.ui.cartao_valor(
                titulo_estimativa, fmt.moeda(final["previsao"]),
                detalhe=f"Faixa provável: {fmt.moeda(final['lim_inf'])} a {fmt.moeda(final['lim_sup'])}",
                fonte=f"Rede LSTM · {self.ui.variacao(final['variacao_pct'])} em relação a hoje"))
        self.ui.renderizar(self.ui.grade(cartoes, colunas=3))

    def _historico(self, alertas: pd.DataFrame) -> None:
        self.ui.renderizar(self.ui.titulo_secao("O que aconteceu nas últimas semanas?"))
        recentes = alertas.head(3)
        if recentes.empty:
            st.info("O sistema não emitiu alertas de compra ou venda para esta ação no período analisado.")
            return
        with st.container(border=True):
            for _, alerta in recentes.iterrows():
                tipo = "compra" if alerta["tipo"] == "compra" else "venda"
                frase = (f"**{fmt.data_curta(alerta['data'])}:** sinal de {tipo} com a ação a "
                         f"{fmt.moeda(alerta['preco'])}.")
                if alerta["resultado"] == "Aguardando":
                    frase += f" Ainda faltam pregões para completar os {self.ctx.horizonte} dias de avaliação."
                else:
                    resultado = ("✔ o sinal acertou a direção" if alerta["resultado"] == "Acerto"
                                 else "✖ o sinal errou a direção")
                    frase += (f" {self.ctx.horizonte} pregões depois ela valia {fmt.moeda(alerta['preco_final'])} "
                              f"({fmt.percentual(alerta['retorno_pct'])}): {resultado}.")
                st.markdown(fmt.escapar_markdown(frase))
        self.botao_navegacao("Ver o histórico completo no Centro de Alertas", "centro_alertas",
                             ":material/notifications:")
