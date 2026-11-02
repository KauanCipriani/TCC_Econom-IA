"""
Painel Avançado: análise técnica completa de um ativo.

Mostra o sinal do modelo escolhido com as probabilidades de cada classe, a
opinião dos quatro modelos (consenso), os indicadores técnicos com a sua
leitura, o gráfico de preço com os alertas emitidos e a previsão da LSTM com
faixa de confiança, os cenários pessimista, esperado e otimista e o que mais
pesou na decisão do modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from econom_ia.config import ATIVOS, CLASSES
from econom_ia.interface import formatacao as fmt
from econom_ia.interface.componentes import ICONES_SINAL
from econom_ia.interface.telas.base import Tela


class TelaPainelAvancado(Tela):
    chave = "painel_avancado"
    rotulo = "Painel Avançado"
    icone = ":material/query_stats:"
    SECOES_OPCIONAIS = {
        "consenso": "O que dizem os quatro modelos",
        "numeros": "Números do dia",
        "grafico": "Gráfico de preço",
        "cenarios": "Cenários de preço da LSTM",
        "indicadores": "Indicadores técnicos",
        "explicacao": "O que mais pesou no sinal",
    }

    def renderizar(self) -> None:
        self.cabecalho("Painel Avançado", "Indicadores técnicos, sinais dos quatro modelos e previsão de preço da LSTM")
        if self.aviso_sem_modelo():
            return

        coluna_ativo, coluna_modelo, coluna_dias = st.columns([1.3, 1.2, 1])
        with coluna_ativo:
            ticker = self.seletor_ativo("Ação", formato_nome_primeiro=False)
        with coluna_modelo:
            chave_modelo = self.seletor_modelo()
        with coluna_dias:
            dias = st.slider("Pregões de previsão", 1, self.ctx.servico_previsao.horizonte_maximo, 5,
                             key="dias_avancado", help="Horizonte da previsão de preço da rede LSTM.")

        dados = self.ctx.ativo(ticker, chave_modelo)
        if dados.empty:
            st.warning(f"Não há cotações disponíveis para {ticker}.")
            return
        ultimo = dados.iloc[-1]
        previsao = self.ctx.servico_previsao.prever(dados, dias)
        if self.ctx.servico_previsao.erro:
            st.info(self.ctx.servico_previsao.erro)

        # 1) O sinal: a resposta e a opinião dos outros modelos.
        self._banner(ticker, dados, chave_modelo, previsao, dias)
        if self.mostrar("consenso"):
            self._consenso(ticker, chave_modelo)
        # 2) O preço: números do dia, gráfico e cenários da LSTM.
        if self.mostrar("numeros") or self.mostrar("grafico"):
            self.ui.renderizar(self.ui.titulo_secao(f"Preço de {ticker}", ATIVOS[ticker].nome))
        if self.mostrar("numeros"):
            self._metricas(dados, previsao, dias)
        if self.mostrar("grafico"):
            self._grafico(ticker, dados, previsao, chave_modelo)
        if self.mostrar("cenarios"):
            self._cenarios(ultimo, previsao, dias)
        # 3) Os porquês: indicadores técnicos e o que mais pesou no sinal.
        if self.mostrar("indicadores"):
            self._indicadores(dados)
        if self.mostrar("explicacao"):
            self._explicacao(dados, chave_modelo)
        self.ui.renderizar(self.ui.aviso_cvm())

    # ── Banner e métricas ──────────────────────────────────────────────────
    def _banner(self, ticker: str, dados: pd.DataFrame, chave_modelo: str, previsao: pd.DataFrame,
                dias: int) -> None:
        ultimo = dados.iloc[-1]
        sinal = ultimo["sinal"] if ultimo["sinal"] in ICONES_SINAL else "neutro"
        partes = []
        if chave_modelo == "lstm":
            partes.append(self._texto_lstm(dados))
        elif not fmt.ausente(ultimo["confianca"]):
            probabilidades = " · ".join(
                f"{ICONES_SINAL[classe]} {classe} {fmt.numero(ultimo[f'p_{classe}'] * 100, 0)}%" for classe in CLASSES)
            if chave_modelo == "svm":
                # O SVM mede distâncias à fronteira de decisão; a normalização
                # (softmax) gera pontuações que somam 100%, mas não são calibradas.
                partes.append(f"Pontuações relativas de cada classe: {probabilidades} (o SVM não estima "
                              "probabilidades calibradas).")
            else:
                partes.append(f"Probabilidades estimadas: {probabilidades}.")

        regra = self.ctx.servico_sinais.regra_limiar
        limiar = self.ctx.servico_sinais.limiar_atual(ultimo) * 100
        detalhe_limiar = "" if regra.tipo == "fixo" else f" (hoje, ±{fmt.numero(limiar)}%)"
        partes.append(f"Movimento relevante: variação maior que {regra.descricao()}{detalhe_limiar} em "
                      f"{self.ctx.horizonte} pregões; abaixo disso a classe é neutra.")
        if not previsao.empty:
            final = previsao.iloc[-1]
            partes.append(f"Previsão da LSTM para {dias} pregões: <strong>{fmt.moeda(final['previsao'])}</strong> "
                          f"({self.ui.variacao(final['variacao_pct'])}).")

        titulo = (f"Sinal do modelo {self.ctx.nome_modelo(chave_modelo)} para {ticker} · "
                  f"pregão de {fmt.data_curta(ultimo['data'])}")
        destaque = f"{ICONES_SINAL[sinal]} {sinal.upper()}"
        if not fmt.ausente(ultimo["confianca"]):
            destaque += f' <span class="banner-confianca">confiança de {fmt.numero(ultimo["confianca"], 0)}%</span>'
        self.ui.renderizar(self.ui.banner(sinal, titulo, destaque, "<br>".join(partes)))

    def _texto_lstm(self, dados: pd.DataFrame) -> str:
        """A LSTM classifica pelo retorno previsto, e não por probabilidades."""
        hiper = (((self.ctx.metricas or {}).get("modelos") or {}).get("lstm") or {}).get("hiperparametros", {})
        previsto = self.ctx.servico_previsao.prever(dados, self.ctx.horizonte)
        if previsto.empty or "corte_venda" not in hiper:
            return "A LSTM classifica pelo retorno que prevê para os próximos pregões."
        return (f"A LSTM não estima probabilidades: o sinal compara o retorno previsto para {self.ctx.horizonte} "
                f"pregões ({fmt.percentual(previsto['variacao_pct'].iloc[-1])}) com os cortes calibrados no "
                f"treinamento ({fmt.percentual(hiper['corte_venda'] * 100)} e "
                f"{fmt.percentual(hiper['corte_compra'] * 100)}).")

    def _metricas(self, dados: pd.DataFrame, previsao: pd.DataFrame, dias: int) -> None:
        ultimo = dados.iloc[-1]
        anterior = dados.iloc[-2] if len(dados) > 1 else ultimo
        mes = dados.iloc[-22] if len(dados) > 22 else dados.iloc[0]
        volume_medio = dados["volume"].iloc[-21:-1].mean()

        variacao_dia = (ultimo["close"] / anterior["close"] - 1) * 100
        variacao_mes = (ultimo["close"] / mes["close"] - 1) * 100
        variacao_volume = (ultimo["volume"] / volume_medio - 1) * 100 if volume_medio else None
        cartoes = [
            self.ui.cartao_valor("Preço atual", fmt.moeda(ultimo["close"]),
                                 detalhe=f"{self.ui.variacao(variacao_dia)} no dia"),
            self.ui.cartao_valor("Variação em 21 pregões", self.ui.variacao(variacao_mes),
                                 detalhe="cerca de um mês de negociação"),
            self.ui.cartao_valor("Volume do dia", fmt.numero_compacto(ultimo["volume"]),
                                 detalhe=fmt.AUSENTE if variacao_volume is None else
                                 f"{fmt.percentual(variacao_volume, 0)} em relação à média de 20 pregões"),
        ]
        if previsao.empty:
            cartoes.append(self.ui.cartao_valor(f"Previsão para {dias} pregões", fmt.AUSENTE))
        else:
            final = previsao.iloc[-1]
            cartoes.append(self.ui.cartao_valor(f"Previsão para {dias} pregões", fmt.moeda(final["previsao"]),
                                                detalhe=f"{self.ui.variacao(final['variacao_pct'])} pela rede LSTM"))
        self.ui.renderizar(self.ui.grade(cartoes, colunas=4))

    # ── Consenso dos modelos ───────────────────────────────────────────────
    def _consenso(self, ticker: str, chave_escolhida: str) -> None:
        modelos = self.ctx.modelos_disponiveis()
        if len(modelos) < 2:
            return
        resultados = self.ctx.metricas.get("modelos", {}) if self.ctx.metricas else {}
        contagem = {classe: 0 for classe in CLASSES}
        cartoes = []
        for chave, nome in modelos.items():
            ultimo = self.ctx.ativo(ticker, chave).iloc[-1]
            sinal = ultimo["sinal"] if ultimo["sinal"] in contagem else "neutro"
            contagem[sinal] += 1
            confianca = ("sem probabilidade" if fmt.ausente(ultimo["confianca"])
                         else f"confiança de {fmt.numero(ultimo['confianca'], 0)}%")
            f1 = ((resultados.get(chave) or {}).get("teste") or {}).get("f1_macro")
            marcas = []
            if chave == self.ctx.modelo_principal:
                marcas.append(self.ui.etiqueta("principal", "info"))
            if chave == chave_escolhida:
                marcas.append(self.ui.etiqueta("em exibição", "aviso"))
            cartoes.append(
                '<div class="cartao">'
                f'<div class="indicador-nome">{nome}</div>'
                f'<div class="consenso-marcas">{" ".join(marcas)}</div>'
                f'<div class="consenso-sinal">{self.ui.etiqueta_sinal(sinal)}</div>'
                f'<div class="indicador-desc">{confianca}<br>F1 no teste: {fmt.numero(f1, 3)}</div></div>')
        maioria = max(contagem, key=contagem.get)
        self.ui.renderizar(self.ui.titulo_secao(
            f"O que dizem os quatro modelos: {contagem[maioria]} de {len(modelos)} indicam {maioria}",
            "modelos diferentes discordarem é esperado quando nenhum deles tem vantagem clara sobre o acaso"))
        self.ui.renderizar(self.ui.grade(cartoes, colunas=min(len(cartoes), 4)))

    # ── Indicadores técnicos ───────────────────────────────────────────────
    def _indicadores(self, dados: pd.DataFrame) -> None:
        self.ui.renderizar(self.ui.titulo_secao("Indicadores técnicos",
                                                "o valor de hoje e a leitura tradicional de cada indicador"))
        cartoes = [self.ui.cartao_indicador(*cartao) for cartao in self.cartoes_indicadores(dados)]
        self.ui.renderizar(self.ui.grade(cartoes, colunas=4))

    @staticmethod
    def cartoes_indicadores(dados: pd.DataFrame) -> list[tuple[str, str, str, str, str]]:
        """(nome, valor, descrição, estado, leitura) de cada indicador no último pregão."""
        u = dados.iloc[-1]
        volume_medio = dados["volume"].iloc[-21:-1].mean()
        cartoes = []

        rsi = u["rsi_14"]
        estado, leitura = (("venda", "▼ Sobrecompra") if rsi > 70 else ("compra", "▲ Sobrevenda") if rsi < 30
                           else ("neutro", "● Neutro"))
        cartoes.append(("RSI (14)", fmt.numero(rsi, 1), "Acima de 70 indica sobrecompra; abaixo de 30, sobrevenda.",
                        estado, leitura))

        hist = u["macd_hist"]
        cartoes.append(("MACD (histograma)", fmt.numero(hist, 3), "Positivo quando o MACD está acima da sua linha "
                        "de sinal (momento de alta).", "compra" if hist > 0 else "venda", None))

        posicao = u["bb_position"]
        estado, leitura = (("venda", "▼ Perto do topo") if posicao > 0.8 else ("compra", "▲ Perto do fundo")
                           if posicao < 0.2 else ("neutro", "● No meio"))
        cartoes.append(("Bandas de Bollinger", f"{fmt.numero(posicao * 100, 0)}%", "Posição do preço entre a banda "
                        "inferior (0%) e a superior (100%).", estado, leitura))

        for periodo in (21, 50):
            distancia = u[f"sma_{periodo}_ratio"] * 100
            cartoes.append((f"Média móvel de {periodo}", fmt.percentual(distancia),
                            f"Distância do preço até a média de {periodo} pregões ({fmt.moeda(u[f'sma_{periodo}'])}).",
                            "compra" if distancia > 0 else "venda", None))

        vol = u["vol_rel_20"]
        estado, leitura = (("aviso", "▲ Acima do normal") if vol > 1.25 else ("info", "▼ Abaixo do normal")
                           if vol < 0.8 else ("neutro", "● Normal"))
        cartoes.append(("Volatilidade relativa", f"{fmt.numero(vol)}x", "Volatilidade de 20 pregões dividida pela "
                        "de 120 pregões.", estado, leitura))

        razao = u["volume"] / volume_medio if volume_medio else np.nan
        estado, leitura = (("info", "▲ Acima da média") if razao > 1.5 else ("neutro", "▼ Abaixo da média")
                           if razao < 0.67 else ("neutro", "● Normal"))
        cartoes.append(("Volume", f"{fmt.numero(razao)}x", "Volume do dia dividido pela média dos 20 pregões "
                        "anteriores.", estado, leitura))

        retorno = (np.exp(u["ret_5"]) - 1) * 100
        cartoes.append(("Retorno em 5 pregões", fmt.percentual(retorno), "Variação acumulada do preço na última "
                        "semana de negociação.", "compra" if retorno > 0 else "venda",
                        "▲ Subindo" if retorno > 0 else "▼ Caindo"))
        return cartoes

    # ── Gráfico e cenários ─────────────────────────────────────────────────
    def _grafico(self, ticker: str, dados: pd.DataFrame, previsao: pd.DataFrame, chave_modelo: str) -> None:
        coluna_periodo, coluna_bandas, coluna_alertas = st.columns([2.2, 1, 1])
        with coluna_periodo:
            pregoes = self.seletor_periodo()
        with coluna_bandas:
            bandas = st.checkbox("Bandas de Bollinger", value=True, key="bandas_avancado")
        with coluna_alertas:
            mostrar_alertas = st.checkbox("Alertas emitidos", value=True, key="alertas_avancado")
        alertas = None
        if mostrar_alertas:
            alertas = self.ctx.alertas(chave_modelo)
            alertas = alertas[alertas["ticker"] == ticker]
        self.exibir(self.graficos.preco_com_sinais(dados, previsao, ticker, pregoes, alertas, bandas),
                    "grafico_avancado")
        st.caption("Triângulos: alertas de compra (▲) e de venda (▼) emitidos pelo modelo escolhido quando o sinal "
                   "mudou. Linha tracejada: previsão da LSTM, com a faixa em que o preço deve ficar com 80% de "
                   "probabilidade, calibrada com os erros da rede no conjunto de teste e ajustada à volatilidade "
                   "atual do ativo.")

    def _cenarios(self, ultimo: pd.Series, previsao: pd.DataFrame, dias: int) -> None:
        if previsao.empty:
            return
        self.ui.renderizar(self.ui.titulo_secao(f"Cenários para daqui a {dias} pregões",
                                                f"até {fmt.data_curta(previsao['data'].iloc[-1])}"))
        final = previsao.iloc[-1]
        atual = float(ultimo["close"])
        cenarios = (
            ("Cenário pessimista", final["lim_inf"], "pessimista", "limite inferior da faixa de 80%"),
            ("Cenário esperado", final["previsao"], "esperado", "previsão central da LSTM"),
            ("Cenário otimista", final["lim_sup"], "otimista", "limite superior da faixa de 80%"),
        )
        self.ui.renderizar(self.ui.grade([
            self.ui.cartao_cenario(titulo, fmt.moeda(valor), self.ui.variacao((valor / atual - 1) * 100), tipo, detalhe)
            for titulo, valor, tipo, detalhe in cenarios], colunas=3))
        with st.expander("Previsão dia a dia"):
            tabela = pd.DataFrame({
                "Pregão": previsao["data"].dt.strftime("%d/%m/%Y"),
                "Previsão (R$)": previsao["previsao"],
                "Limite inferior (R$)": previsao["lim_inf"],
                "Limite superior (R$)": previsao["lim_sup"],
                "Variação prevista (%)": previsao["variacao_pct"],
            })
            reais = fmt.formatador(2)
            self.tabela(tabela, {"Previsão (R$)": reais, "Limite inferior (R$)": reais,
                                 "Limite superior (R$)": reais, "Variação prevista (%)": fmt.formatador(2, sinal=True)})

    # ── Explicação do sinal ────────────────────────────────────────────────
    def _explicacao(self, dados: pd.DataFrame, chave_modelo: str) -> None:
        self.ui.renderizar(self.ui.titulo_secao("O que mais pesou no sinal"))
        if chave_modelo == "lstm":
            self.ui.renderizar(self.ui.nota(
                "A LSTM analisa a sequência dos últimos pregões como um todo, por isso a contribuição de cada "
                "indicador isolado não é calculada para ela. Escolha outro modelo para ver a explicação."))
            return
        contribuicoes = self.ctx.servico_sinais.explicar(dados, chave_modelo)
        if not contribuicoes:
            st.info("Não foi possível calcular a explicação para este sinal.")
            return
        sinal = dados.iloc[-1]["sinal"]
        principais = sorted(contribuicoes, key=lambda c: -abs(c["contribuicao"]))[:8]
        principais.sort(key=lambda c: -c["contribuicao"])
        maior = contribuicoes[0]
        # O SVM gera pontuações relativas (softmax), e não probabilidades estimadas.
        medida = "pontuação" if chave_modelo == "svm" else "probabilidade"
        self.ui.renderizar(self.ui.nota(
            f"Para cada indicador, o sistema troca o valor de hoje pelo valor típico do ativo (a mediana do "
            f"histórico) e mede quanto a {medida} de <strong>{str(sinal).upper()}</strong> muda. O que mais "
            f"pesou foi <strong>{maior['descricao']}</strong>: sem ele, a {medida} cairia "
            f"{fmt.numero(maior['contribuicao'], 1)} pontos percentuais. Valores negativos indicam indicadores que "
            f"pesaram contra o sinal."))
        self.exibir(self.graficos.barras_simples(
            [c["descricao"] for c in principais], [c["contribuicao"] for c in principais],
            "Contribuição (pontos percentuais)", formato="{:+.1f}", horizontal=True, polaridade=True),
            "grafico_explicacao")
        st.caption("Barras azuis: indicadores que reforçam o sinal. Barras cinzas: indicadores que pesam contra ele.")
