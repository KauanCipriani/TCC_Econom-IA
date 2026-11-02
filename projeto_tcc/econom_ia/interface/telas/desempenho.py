"""
Tela Desempenho dos Modelos: a avaliação feita no treinamento, com transparência.

Lê o arquivo de métricas gerado por treinar_modelos.py e mostra: a comparação
dos quatro modelos com as referências (classificador aleatório, classe
majoritária e o Random Forest do TC I), com intervalos de confiança; os
experimentos de limiar e de conjuntos de variáveis; a qualidade da previsão de
preço da LSTM frente à previsão ingênua; e a simulação de carteira.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from econom_ia.config import ATIVOS, CLASSES
from econom_ia.features import CalculadoraIndicadores
from econom_ia.interface import dados_app
from econom_ia.interface import formatacao as fmt
from econom_ia.interface.graficos import NOMES_CONJUNTOS
from econom_ia.interface.telas.base import Tela

MODELOS = ("random_forest", "svm", "mlp", "lstm")


class TelaDesempenho(Tela):
    chave = "desempenho"
    rotulo = "Desempenho dos Modelos"
    icone = ":material/analytics:"
    SECOES_OPCIONAIS = {"informacoes": "Períodos, variável-alvo e detalhes do treinamento"}

    # Hiperparâmetros mostrados no detalhe de cada modelo.
    HIPERPARAMETROS_EXIBIDOS = {"max_depth", "min_samples_leaf", "max_features", "n_estimators", "C", "gamma",
                                "camadas", "dropout", "decaimento_peso", "janela", "oculto", "epocas_treinadas"}

    def renderizar(self) -> None:
        self.cabecalho("Desempenho dos Modelos",
                       "Como os modelos se saíram em dados que não foram usados no treinamento")
        metricas = self.ctx.metricas
        if not metricas:
            self.aviso_sem_modelo()
            return
        self._resumo(metricas)
        if self.mostrar("informacoes"):
            self._informacoes(metricas)
        abas = st.tabs(["Classificação", "Limiar e variáveis", "Previsão de preço (LSTM)", "Simulação de carteira"])
        with abas[0]:
            self._classificacao(metricas)
        with abas[1]:
            self._limiar_e_variaveis(metricas)
        with abas[2]:
            self._previsao_preco(metricas)
        with abas[3]:
            self._simulacao(metricas)

    # ── Resposta principal ─────────────────────────────────────────────────
    def _resumo(self, metricas: dict) -> None:
        """Resultado em uma frase: o modelo principal comparado a um palpite aleatório."""
        principal = metricas["modelo_principal"]
        resultados = metricas["modelos"].get(principal) or {}
        aleatorio = metricas.get("referencias", {}).get("stratified", {})
        conjuntos = [("teste", "no teste"), ("oos", "na SANB11, ação nunca vista no treino"),
                     ("prospectivo", "nos pregões posteriores ao estudo")]
        partes = [f"<strong>{fmt.numero(resultados[c]['f1_macro'], 3)}</strong> {texto}"
                  for c, texto in conjuntos if resultados.get(c)]
        referencia = [aleatorio[c]["f1_macro"] for c, _ in conjuntos if aleatorio.get(c)]
        if not partes:
            return
        frase_ref = (f", contra cerca de {fmt.numero(sum(referencia) / len(referencia), 2)} de um palpite "
                     "aleatório" if referencia else "")
        self.ui.renderizar(self.ui.banner(
            "neutro", "Em resumo",
            "Os modelos acertam pouco mais do que um palpite aleatório",
            f"O modelo principal, {self.ctx.nome_modelo(principal)}, obteve F1-Score de "
            f"{', '.join(partes[:-1])} e {partes[-1]}{frase_ref}. Os outros três modelos ficaram na mesma faixa. "
            "Por isso o sistema deve ser usado como mais uma fonte de informação, nunca como a única base para "
            "decidir. As abas abaixo detalham cada experimento."))

    # ── Informações gerais ─────────────────────────────────────────────────
    def _informacoes(self, metricas: dict) -> None:
        info = metricas["info"]
        estudo, prospectivo = info["periodo_estudo"], info["periodo_prospectivo"]
        principal = metricas["modelo_principal"]
        self.ui.renderizar(self.ui.grade([
            self.ui.cartao_valor("Período do estudo", f"{self._mes(estudo[0])} a {self._mes(estudo[1])}",
                                 detalhe="datas com todas as variáveis e com a variável-alvo conhecida", texto=True),
            self.ui.cartao_valor("Período prospectivo", f"{self._mes(prospectivo[0])} a {self._mes(prospectivo[1])}",
                                 detalhe="pregões nunca usados no treinamento nem na escolha dos hiperparâmetros",
                                 texto=True),
            self.ui.cartao_valor("Variável-alvo", f"{info['limiar']['descricao']}",
                                 detalhe=f"variação em {info['horizonte']} pregões que separa compra, neutro e venda",
                                 texto=True),
            self.ui.cartao_valor("Modelo principal", self.ctx.nome_modelo(principal),
                                 detalhe="maior F1-Score médio na validação cruzada temporal", texto=True),
        ]))

        with st.expander("Detalhes do treinamento"):
            registros = info["registros"]
            limpeza = info.get("limpeza", {})
            linhas = [
                ("Data do treinamento", info["data_treinamento"]),
                ("Fonte dos dados", info["fonte_dados"]),
                ("Ações de treinamento", ", ".join(info["tickers_treino"])),
                ("Ação out-of-sample", f"{info['ticker_out_of_sample']} (nunca usada no ajuste dos modelos)"),
                ("Início do conjunto de teste", fmt.data_curta(pd.Timestamp(info["inicio_teste"]))),
                ("Registros (treino / teste / OOS / prospectivo)",
                 f"{registros['treino']} / {registros['teste']} / {registros['oos']} / {registros['prospectivo']}"),
                ("Variáveis de entrada", f"{len(info['features'])}: " + ", ".join(info["features"])),
                ("Validação cruzada", f"{info['n_folds']} partições temporais com expurgo de {info['horizonte']} "
                                      "pregões entre treino e validação"),
                ("Limpeza dos dados", f"{limpeza.get('sem_negociacao_removidos', 0)} pregões sem negociação removidos;"
                                      f" {limpeza.get('outliers_marcados', 0)} variações extremas sinalizadas"),
                ("Tempo total de treinamento", f"{fmt.numero(metricas.get('duracao_total_s'), 0)} segundos"),
                ("Versões", ", ".join(f"{k} {v}" for k, v in info.get("versoes", {}).items() if v)),
            ]
            self.tabela(pd.DataFrame(linhas, columns=["Item", "Valor"]))

    @staticmethod
    def _mes(data_iso: str) -> str:
        return fmt.mes_ano(pd.Timestamp(data_iso))

    # ── Classificação ──────────────────────────────────────────────────────
    def _classificacao(self, metricas: dict) -> None:
        self.ui.renderizar(self.ui.nota(
            "<strong>Teste:</strong> últimos 30% das datas do estudo, nas três ações de treino. "
            "<strong>Out-of-sample:</strong> SANB11 nas mesmas datas, ação que os modelos nunca viram. "
            "<strong>Prospectivo:</strong> pregões após o fim do estudo, com os modelos retreinados em todo o "
            "período. O F1-Score macro dá o mesmo peso às três classes; o Coeficiente de Correlação de Matthews "
            "(MCC) vale 0 para um palpite aleatório e 1 para acerto total. As barras de erro são intervalos de "
            "confiança de 95% obtidos por bootstrap em blocos."))
        modelos = metricas["modelos"]
        referencias = metricas.get("referencias", {})
        linhas = []
        for chave in MODELOS:
            for conjunto in ("teste", "oos", "prospectivo"):
                resultado = (modelos.get(chave) or {}).get(conjunto)
                if resultado:
                    ic = resultado.get("f1_ic95") or [None, None]
                    linhas.append({"modelo": self.ctx.nome_modelo(chave), "conjunto": conjunto,
                                   "f1": resultado["f1_macro"], "ic_inferior": ic[0], "ic_superior": ic[1]})
        aleatorio = ((referencias.get("stratified") or {}).get("teste") or {}).get("f1_macro")
        self.exibir(self.graficos.comparacao_modelos(linhas, aleatorio), "grafico_comparacao")

        conjunto = st.radio("Conjunto de avaliação", list(NOMES_CONJUNTOS), horizontal=True,
                            format_func=NOMES_CONJUNTOS.get, key="conjunto_desempenho")
        tres_casas = fmt.formatador(3)
        self.tabela(self.tabela_classificacao(metricas, conjunto),
                    {coluna: tres_casas for coluna in ("F1 macro", "MCC", "Acurácia", "Precisão", "Revocação")})
        st.caption("Aleatório estratificado: sorteia as classes com as proporções do treino. Classe majoritária: "
                   "prevê sempre a classe mais frequente. TC I: o Random Forest da primeira etapa (limiar de ±1,5% "
                   "e 8 variáveis), reproduzido com os dados reais e avaliado da mesma forma.")

        self._detalhes_modelo(metricas, conjunto)

    def tabela_classificacao(self, metricas: dict, conjunto: str) -> pd.DataFrame:
        """Métricas de todos os modelos e referências em um conjunto de avaliação."""
        candidatos = [(self.ctx.nome_modelo(c), metricas["modelos"].get(c, {}).get(conjunto)) for c in MODELOS]
        for chave, referencia in (metricas.get("referencias") or {}).items():
            candidatos.append((referencia.get("nome", chave), referencia.get(conjunto)))
        candidatos.append(("Random Forest do TC I", (metricas.get("referencia_tc1") or {}).get(conjunto)))

        def intervalo(valores) -> str:
            return fmt.AUSENTE if not valores else f"[{fmt.numero(valores[0], 3)}; {fmt.numero(valores[1], 3)}]"

        linhas = []
        for nome, resultado in candidatos:
            if not resultado:
                continue
            linhas.append({
                "Modelo": nome,
                "F1 macro": resultado["f1_macro"],
                "IC 95% do F1": intervalo(resultado.get("f1_ic95")),
                "MCC": resultado["mcc"],
                "IC 95% do MCC": intervalo(resultado.get("mcc_ic95")),
                "Acurácia": resultado["accuracy"],
                "Precisão": resultado["precision_macro"],
                "Revocação": resultado["recall_macro"],
                "Amostras": resultado["n_amostras"],
            })
        return pd.DataFrame(linhas)

    def _detalhes_modelo(self, metricas: dict, conjunto: str) -> None:
        self.ui.renderizar(self.ui.titulo_secao("Detalhes de um modelo", NOMES_CONJUNTOS[conjunto]))
        disponiveis = [c for c in MODELOS if c in metricas["modelos"]]
        chave = st.selectbox("Modelo", disponiveis, format_func=self.ctx.nome_modelo, key="modelo_detalhe")
        dados = metricas["modelos"][chave]
        resultado = dados.get(conjunto) or {}
        esquerda, direita = st.columns([1.1, 1], gap="large")
        with esquerda:
            if resultado.get("matriz_confusao"):
                self.exibir(self.graficos.matriz_confusao(resultado["matriz_confusao"]), "grafico_matriz")
                st.caption("Cada linha mostra, para uma classe real, quantos casos o modelo previu em cada classe "
                           "e a proporção em relação ao total da linha.")
        with direita:
            por_classe = resultado.get("f1_por_classe", {})
            cv = dados.get("cv", {})
            linhas = [(f"F1 da classe {classe}", fmt.numero(por_classe.get(classe), 3)) for classe in CLASSES]
            linhas += [
                ("F1 na validação cruzada",
                 f"{fmt.numero(cv.get('f1_media'), 3)} ± {fmt.numero(cv.get('f1_desvio'), 3)}"),
                ("Combinações testadas", str(cv.get("combinacoes_testadas", fmt.AUSENTE))),
                ("Tempo de treinamento", f"{fmt.numero(dados.get('tempo_treino_s'), 1)} s"),
            ]
            linhas += [(f"Hiperparâmetro: {nome}", str(valor)) for nome, valor in dados["hiperparametros"].items()
                       if nome in self.HIPERPARAMETROS_EXIBIDOS]
            self.tabela(pd.DataFrame(linhas, columns=["Item", "Valor"]))

        prospectivo = (dados.get("prospectivo") or {}).get("f1_por_ativo")
        if prospectivo:
            tickers = [t for t in ATIVOS if t in prospectivo]
            aleatorio = (((metricas.get("referencias") or {}).get("stratified") or {}).get("prospectivo")
                         or {}).get("f1_macro")
            self.ui.renderizar(self.ui.titulo_secao("F1-Score por ação no período prospectivo"))
            self.exibir(self.graficos.barras_simples(
                tickers, [prospectivo[t] for t in tickers], "F1-Score macro", referencia=aleatorio,
                rotulo_referencia=f"Aleatório estratificado no prospectivo ({fmt.numero(aleatorio, 3)})"),
                "grafico_f1_ativos")
        if dados.get("importancias"):
            importancias = sorted(dados["importancias"].items(), key=lambda item: -item[1])
            self.ui.renderizar(self.ui.titulo_secao("Importância das variáveis", "redução média de impureza (Gini)"))
            self.exibir(self.graficos.barras_simples(
                [CalculadoraIndicadores.DESCRICOES.get(n, n) for n, _ in importancias],
                [v for _, v in importancias], "Importância", horizontal=True), "grafico_importancias")

    # ── Limiar e conjuntos de variáveis ────────────────────────────────────
    def _limiar_e_variaveis(self, metricas: dict) -> None:
        escolhido = metricas["info"]["limiar"]["descricao"]
        self.ui.renderizar(self.ui.nota(
            "Os experimentos usaram um Random Forest de configuração fixa e a validação cruzada temporal apenas "
            "nos dados de treino. O limiar foi escolhido pelo maior MCC médio, que desconta o acerto por acaso, e o "
            f"conjunto de variáveis pelo maior F1-Score. Limiar escolhido: <strong>{escolhido}</strong>."))
        experimento = metricas.get("experimento_limiar", [])
        if experimento:
            self.ui.renderizar(self.ui.titulo_secao("Reavaliação do limiar da variável-alvo"))
            rotulos = [e["limiar"]["descricao"] for e in experimento]
            esquerda, direita = st.columns(2, gap="large")
            with esquerda:
                self.exibir(self.graficos.barras_agrupadas(
                    rotulos, {"Random Forest (validação cruzada)": [e["f1_cv_media"] for e in experimento],
                              "Aleatório estratificado": [e["f1_aleatorio"] for e in experimento]},
                    "F1-Score macro"), "grafico_limiar_f1")
            with direita:
                self.exibir(self.graficos.barras_simples(
                    rotulos, [e["mcc_cv_media"] for e in experimento], "MCC médio",
                    erros=[e["mcc_cv_desvio"] for e in experimento], referencia=0,
                    rotulo_referencia="Acaso (MCC = 0)"),
                    "grafico_limiar_mcc")
            tabela = pd.DataFrame([{
                "Limiar": ("✔ " if e["limiar"]["descricao"] == escolhido else "") + e["limiar"]["descricao"],
                "Venda (%)": e["distribuicao"]["venda"] * 100,
                "Neutro (%)": e["distribuicao"]["neutro"] * 100,
                "Compra (%)": e["distribuicao"]["compra"] * 100,
                "F1 (CV)": e["f1_cv_media"],
                "F1 aleatório": e["f1_aleatorio"],
                "MCC (CV)": e["mcc_cv_media"],
            } for e in experimento])
            self.tabela(tabela, {
                **{c: fmt.formatador(1) for c in ("Venda (%)", "Neutro (%)", "Compra (%)")},
                **{c: fmt.formatador(3) for c in ("F1 (CV)", "F1 aleatório", "MCC (CV)")},
            })
            st.caption("O limiar adaptativo multiplica a volatilidade recente do ativo (20 pregões, projetada para "
                       "5 pregões) por um fator fixo, de modo que o movimento considerado relevante acompanha o "
                       "nível de risco de cada período.")

        variaveis = metricas.get("experimento_features", [])
        if variaveis:
            self.ui.renderizar(self.ui.titulo_secao("Conjuntos de variáveis"))
            nomes = {"originais": "Originais do TC I", "novas": "Novas do TC II", "completas": "Todas"}
            escolhido_vars = max(variaveis, key=lambda v: v["f1_cv_media"])["conjunto"]
            esquerda, direita = st.columns([1.2, 1], gap="large")
            with esquerda:
                self.exibir(self.graficos.barras_simples(
                    [f"{nomes.get(v['conjunto'], v['conjunto'])} ({v['n_features']})" for v in variaveis],
                    [v["f1_cv_media"] for v in variaveis], "F1-Score macro (validação cruzada)",
                    erros=[v["f1_cv_desvio"] for v in variaveis]), "grafico_variaveis")
            with direita:
                self.tabela(pd.DataFrame([{
                    "Conjunto": ("✔ " if v["conjunto"] == escolhido_vars else "")
                                + nomes.get(v["conjunto"], v["conjunto"]),
                    "Variáveis": v["n_features"],
                    "F1 (CV)": v["f1_cv_media"],
                    "Desvio": v["f1_cv_desvio"],
                    "MCC (CV)": v["mcc_cv_media"],
                } for v in variaveis]), {c: fmt.formatador(3) for c in ("F1 (CV)", "Desvio", "MCC (CV)")})
            with st.expander("O que é cada variável"):
                self.tabela(pd.DataFrame(
                    [{"Variável": n, "Descrição": CalculadoraIndicadores.DESCRICOES.get(n, n),
                      "Origem": "TC I" if n in CalculadoraIndicadores.FEATURES_ORIGINAIS else "TC II"}
                     for n in metricas["info"]["features"]]))

    # ── Previsão de preço ──────────────────────────────────────────────────
    def _previsao_preco(self, metricas: dict) -> None:
        regressao = metricas.get("regressao_lstm") or {}
        if not regressao:
            st.info("Não há resultados da LSTM neste treinamento.")
            return
        self.ui.renderizar(self.ui.nota(
            "A referência é a <strong>previsão ingênua</strong>, que supõe que o preço futuro será igual ao de hoje "
            "(passeio aleatório). Ganho positivo significa erro menor que o dela. O acerto de direção é a proporção "
            "de vezes em que a LSTM previu corretamente se o preço subiria ou cairia. A cobertura mostra em quantos "
            "casos o preço real ficou dentro da faixa de 80%."))
        conjunto = st.radio("Conjunto", list(NOMES_CONJUNTOS), horizontal=True, format_func=NOMES_CONJUNTOS.get,
                            key="conjunto_lstm")
        tabela = self.tabela_regressao(regressao.get(conjunto) or {})
        if tabela.empty:
            st.info("Sem resultados para este conjunto.")
        else:
            self.tabela(tabela, {
                "RMSE LSTM (R$)": fmt.formatador(3),
                "RMSE ingênuo (R$)": fmt.formatador(3),
                "Ganho sobre o ingênuo (%)": fmt.formatador(2, sinal=True),
                "MAPE LSTM (%)": fmt.formatador(2),
                "Acerto de direção (%)": fmt.formatador(1),
                "Cobertura faixa fixa (%)": fmt.formatador(1),
                "Cobertura faixa ajustada (%)": fmt.formatador(1),
            })
        if conjunto == "prospectivo":
            st.caption("Faixa fixa: quantis dos erros do teste aplicados sem ajuste. Faixa ajustada: os mesmos erros "
                       "medidos em unidades de volatilidade e reescalados pela volatilidade de cada dia, método usado "
                       "na interface.")

        previsoes = dados_app.previsoes_avaliacao(metricas["info"]["data_treinamento"])
        if previsoes is None or previsoes.empty:
            return
        parte = previsoes[previsoes["conjunto"] == conjunto]
        tickers = [t for t in ATIVOS if t in set(parte["ticker"])]
        if not tickers:
            return
        horizonte = self.ctx.horizonte
        self.ui.renderizar(self.ui.titulo_secao("Previsto versus real", f"{horizonte} pregões à frente"))
        ticker = st.selectbox("Ação", tickers, key="ticker_lstm", format_func=lambda t: f"{t} · {ATIVOS[t].nome}")
        serie = parte[parte["ticker"] == ticker].dropna(subset=["retorno_futuro", "lstm_r_5"]).sort_values("data")
        comparacao = pd.DataFrame({
            "data": serie["data"],
            "preco_real": serie["close"] * (1 + serie["retorno_futuro"]),
            "preco_previsto": serie["close"] * np.exp(serie["lstm_r_5"].astype(float)),
        })
        self.exibir(self.graficos.previsto_vs_real(comparacao, ticker, horizonte), "grafico_previsto_real")
        st.caption("As duas linhas quase se sobrepõem porque o preço de hoje já explica a maior parte do preço de "
                   "daqui a poucos dias; o que importa é se a LSTM erra menos do que a previsão ingênua.")

        grade = (metricas["modelos"].get("lstm") or {}).get("grade_lstm")
        if grade:
            with st.expander("Escolha da arquitetura da LSTM"):
                self.tabela(pd.DataFrame([{
                    "Janela (pregões)": g["janela"], "Neurônios ocultos": g["oculto"],
                    "Perda de validação": g["perda_validacao"], "Épocas treinadas": g["epocas_treinadas"],
                } for g in grade]), {"Perda de validação": fmt.formatador(4)})

    @staticmethod
    def tabela_regressao(resultados: dict) -> pd.DataFrame:
        linhas = []
        for horizonte, r in sorted(resultados.items(), key=lambda item: int(item[0])):
            linha = {
                "Horizonte (pregões)": int(horizonte),
                "RMSE LSTM (R$)": r["lstm"]["rmse"],
                "RMSE ingênuo (R$)": r["ingenuo"]["rmse"],
                "Ganho sobre o ingênuo (%)": r["ganho_rmse_vs_ingenuo"] * 100,
                "MAPE LSTM (%)": r["lstm"]["mape"],
                "Acerto de direção (%)": r["lstm"]["acerto_direcao"] * 100,
            }
            if "cobertura_faixa_fixa" in r:
                linha["Cobertura faixa fixa (%)"] = r["cobertura_faixa_fixa"] * 100
                linha["Cobertura faixa ajustada (%)"] = r["cobertura_faixa_volatilidade"] * 100
            linhas.append(linha)
        return pd.DataFrame(linhas)

    # ── Simulação de carteira ──────────────────────────────────────────────
    def _simulacao(self, metricas: dict) -> None:
        backtest = metricas.get("backtest") or {}
        if not backtest:
            st.info("Não há simulação neste treinamento.")
            return
        self.ui.renderizar(self.ui.nota(
            "Regra simulada: ao fim de cada pregão, a carteira fica comprada na ação quando o sinal é "
            "<strong>compra</strong> e fora do mercado nos demais casos, com o mesmo peso para cada ação e custo de "
            "0,05% a cada entrada ou saída. A referência é comprar e manter as mesmas ações. Rendimento do caixa, "
            "impostos e deslizamento de preço não são considerados: a simulação verifica se os sinais têm valor "
            "econômico e não é promessa de rentabilidade."))
        opcoes = [c for c in ("teste", "prospectivo") if c in backtest]
        conjunto = st.radio("Período", opcoes, horizontal=True, format_func=NOMES_CONJUNTOS.get,
                            key="conjunto_backtest")
        resultado = backtest[conjunto]
        nomes = {c: self.ctx.nome_modelo(c) for c in MODELOS if c in resultado}
        linhas = []
        for chave, nome in [*nomes.items(), ("comprar_e_manter", "Comprar e manter")]:
            r = resultado[chave]
            linhas.append({
                "Estratégia": nome,
                "Retorno total (%)": r["retorno_total"] * 100,
                "Retorno anualizado (%)": r["retorno_anualizado"] * 100,
                "Volatilidade anual (%)": r["volatilidade_anual"] * 100,
                "Índice de Sharpe": r["indice_sharpe"],
                "Perda máxima (%)": r["max_drawdown"] * 100,
                "Tempo comprado (%)": r["exposicao_media"] * 100,
                "Operações": r["n_operacoes"],
            })
        self.tabela(pd.DataFrame(linhas), {
            **{c: fmt.formatador(1, sinal=True) for c in
               ("Retorno total (%)", "Retorno anualizado (%)", "Perda máxima (%)")},
            "Volatilidade anual (%)": fmt.formatador(1),
            "Índice de Sharpe": fmt.formatador(2),
            "Tempo comprado (%)": fmt.formatador(0),
        })
        st.caption("Índice de Sharpe calculado sem taxa livre de risco (retorno médio anualizado dividido pela "
                   "volatilidade anualizada). Perda máxima: maior queda do valor da carteira a partir de um pico.")
        if "curvas" in resultado:
            self.exibir(self.graficos.curvas_capital(resultado["curvas"], nomes), "grafico_curvas")
