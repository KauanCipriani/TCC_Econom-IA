"""
Pipeline completo de experimentos e treinamento (TC II).

Etapas executadas por PipelineTreinamento.executar():

  1. Carrega o OHLCV, trata ausentes, marca outliers e calcula os indicadores.
  2. Separa o período do estudo (jan/2019 a set/2025) do período prospectivo
     (pregões posteriores, usados só para avaliação) e divide o estudo em
     treino e teste (70/30 por data, com expurgo). A SANB11 fica fora do treino.
  3. Reavalia o limiar da variável-alvo e compara conjuntos de variáveis.
  4. Ajusta os hiperparâmetros de Random Forest, SVM e MLP por busca em grade
     e os da LSTM em uma validação separada, sempre com validação temporal.
  5. Avalia todos os modelos no teste, na SANB11 e no período prospectivo,
     com intervalos de confiança, comparação com referências ingênuas e uma
     simulação de estratégia.
  6. Retreina os modelos com todo o período do estudo e grava modelos,
     métricas e bases para a interface.
"""

from __future__ import annotations

import copy
import json
import logging
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, matthews_corrcoef

from econom_ia.avaliacao import (
    AvaliadorClassificacao,
    AvaliadorRegressao,
    Backtester,
    DivisorTemporalPurgado,
    dividir_por_data,
)
from econom_ia.config import CAMINHOS, CLASSES, EXPERIMENTO, Caminhos, ConfigExperimento
from econom_ia.dados import LimpadorDados, RepositorioDados
from econom_ia.features import CalculadoraIndicadores, RegraLimiar, Rotulador
from econom_ia.modelos import (
    ModeloLSTMClassificacao,
    ModeloMLP,
    ModeloRandomForest,
    ModeloSVM,
    PrevisorLSTM,
)
from econom_ia.modelos.base import ModeloClassificacao
from econom_ia.modelos.random_forest import ModeloRandomForestTC1

log = logging.getLogger("econom_ia.treinamento")

# Hiperparâmetros fixos do Random Forest usados nos experimentos de limiar e
# de variáveis (antes da busca em grade), para isolar o efeito estudado.
PARAMETROS_EXPERIMENTO_RF = {"max_depth": 8, "min_samples_leaf": 30, "max_features": "sqrt"}

# Grade da LSTM (avaliada em validação separada, por custo computacional).
GRADE_LSTM = [
    {"janela": 20, "oculto": 32},
    {"janela": 20, "oculto": 64},
    {"janela": 40, "oculto": 32},
    {"janela": 40, "oculto": 64},
]

HORIZONTES_RELATORIO = (1, 5, 10, 15)


class PipelineTreinamento:
    """Orquestra coleta local, experimentos, avaliação e persistência."""

    def __init__(
        self,
        config: ConfigExperimento = EXPERIMENTO,
        caminhos: Caminhos = CAMINHOS,
        modo_rapido: bool = False,
    ) -> None:
        """
        Args:
            modo_rapido: reduz grades e épocas (usado nos testes automatizados).
        """
        self.config = config
        self.caminhos = caminhos
        self.modo_rapido = modo_rapido
        self.repositorio = RepositorioDados(caminhos)
        self.indicadores = CalculadoraIndicadores()
        self.avaliador = AvaliadorClassificacao()
        self.divisor = DivisorTemporalPurgado(config.n_folds, config.horizonte)
        self.metricas: dict = {}
        self._base: pd.DataFrame | None = None  # indicadores de todas as datas

    # ══════════════════════════════════════════════════════════════════════
    # Execução
    # ══════════════════════════════════════════════════════════════════════
    def executar(self) -> dict:
        inicio = time.perf_counter()
        cfg = self.config

        log.info("[1/6] Preparando a base de dados")
        base, relatorio_limpeza = self._preparar_base()
        self._base = base

        log.info("[2/6] Separando períodos e conjuntos")
        conjuntos = self._montar_conjuntos(base, RegraLimiar(*cfg.limiar_padrao))
        treino = conjuntos["treino"]

        log.info("[3/6] Experimentos de limiar e de variáveis")
        experimento_limiar, limiar = self._experimento_limiar(treino)
        conjuntos = self._montar_conjuntos(base, limiar)
        treino, teste, prospectivo = conjuntos["treino"], conjuntos["teste"], conjuntos["prospectivo"]
        experimento_features, features = self._experimento_features(treino)
        referencia_tc1 = self._reproduzir_tc1(base)

        log.info("[4/6] Ajuste de hiperparâmetros e treinamento")
        modelos = self._treinar_modelos(treino, features)

        log.info("[5/6] Avaliação nos conjuntos de teste, out-of-sample e prospectivo")
        avaliacao = self._avaliar_modelos(modelos, conjuntos)
        regressao = self._avaliar_regressao(modelos["lstm"].previsor, conjuntos)
        referencias = self._avaliar_referencias(treino, conjuntos)
        backtest_teste = self._backtest(modelos, teste)

        log.info("[6/6] Retreinando com todo o período do estudo e salvando")
        producao = self._treinar_producao(modelos, conjuntos["estudo"])
        avaliacao_prospectiva = self._avaliar_prospectivo(producao, prospectivo)
        regressao["prospectivo"] = self._avaliar_regressao_prospectiva(
            producao["lstm"].previsor, prospectivo, regressao["faixas"])
        backtest_prospectivo = self._backtest(producao, prospectivo)

        principal = max(
            ("random_forest", "svm", "mlp", "lstm"),
            key=lambda chave: avaliacao[chave]["cv"]["f1_media"],
        )
        self.metricas = {
            "info": self._informacoes(conjuntos, limiar, features, relatorio_limpeza, principal),
            "experimento_limiar": experimento_limiar,
            "experimento_features": experimento_features,
            "referencia_tc1": referencia_tc1,
            "modelos": {
                chave: {**avaliacao[chave], "prospectivo": avaliacao_prospectiva[chave]}
                for chave in avaliacao
            },
            "referencias": referencias,
            "regressao_lstm": regressao,
            "backtest": {"teste": backtest_teste, "prospectivo": backtest_prospectivo},
            "modelo_principal": principal,
            "duracao_total_s": round(time.perf_counter() - inicio, 1),
        }
        self._salvar(producao, conjuntos, modelos)
        log.info("Treinamento concluído em %.1f s. Modelo principal: %s",
                 self.metricas["duracao_total_s"], principal)
        return self.metricas

    # ══════════════════════════════════════════════════════════════════════
    # Preparação dos dados
    # ══════════════════════════════════════════════════════════════════════
    def _preparar_base(self) -> tuple[pd.DataFrame, dict]:
        ohlcv = self.repositorio.carregar_ohlcv()
        limpo, relatorio = LimpadorDados().limpar(ohlcv)
        log.info("  %d registros; %d removidos sem negociação; %d outliers marcados",
                 relatorio.registros_saida, relatorio.sem_negociacao_removidos,
                 relatorio.outliers_marcados)
        return self.indicadores.calcular(limpo), relatorio.como_dict()

    def _montar_conjuntos(self, base: pd.DataFrame, limiar: RegraLimiar) -> dict[str, pd.DataFrame]:
        """Rotula a base e forma os conjuntos de treino, teste, OOS e prospectivo."""
        cfg = self.config
        colunas = CalculadoraIndicadores.FEATURES_COMPLETAS
        rotulador = Rotulador(cfg.horizonte, limiar)
        fim_estudo = pd.Timestamp(cfg.fim_estudo)

        # No estudo, o alvo usa apenas preços do próprio período.
        estudo = base[(base["data"] >= pd.Timestamp(cfg.inicio_estudo)) & (base["data"] <= fim_estudo)]
        estudo = rotulador.adicionar_alvo(estudo).dropna(subset=[*colunas, "alvo"])
        estudo = estudo.sort_values(["data", "ticker"]).reset_index(drop=True)

        tres = estudo[estudo["ticker"].isin(cfg.tickers_treino)]
        treino, teste, inicio_teste = dividir_por_data(tres, cfg.proporcao_treino, cfg.horizonte)
        oos = estudo[(estudo["ticker"] == cfg.ticker_oos) & (estudo["data"] >= inicio_teste)]
        oos_completo = estudo[estudo["ticker"] == cfg.ticker_oos]

        prospectivo = rotulador.adicionar_alvo(base)
        prospectivo = prospectivo[prospectivo["data"] > fim_estudo].dropna(subset=[*colunas, "alvo"])
        prospectivo = prospectivo.sort_values(["data", "ticker"]).reset_index(drop=True)

        return {
            "estudo": estudo,
            "estudo_treino_modelos": tres,
            "treino": treino.reset_index(drop=True),
            "teste": teste.reset_index(drop=True),
            "oos": oos.reset_index(drop=True),
            "oos_completo": oos_completo.reset_index(drop=True),
            "prospectivo": prospectivo,
            "inicio_teste": inicio_teste,
            "limiar": limiar,
        }

    def _rotular_novamente(self, df: pd.DataFrame, limiar: RegraLimiar) -> pd.DataFrame:
        rotulador = Rotulador(self.config.horizonte, limiar)
        limiares = rotulador.limiares(df)
        return df.assign(limiar_alvo=limiares, alvo=rotulador.classificar(df["retorno_futuro"], limiares))

    # ══════════════════════════════════════════════════════════════════════
    # Validação cruzada
    # ══════════════════════════════════════════════════════════════════════
    def _prever_alinhado(self, modelo: ModeloClassificacao, alvo: pd.DataFrame) -> np.ndarray:
        """Previsões para as linhas de "alvo"; a LSTM recebe o histórico anterior."""
        if not isinstance(modelo, ModeloLSTMClassificacao):
            return modelo.prever(alvo)
        historico = self._base[
            self._base["ticker"].isin(alvo["ticker"].unique())
            & (self._base["data"] <= alvo["data"].max())
        ]
        previsto = pd.DataFrame({
            "ticker": historico["ticker"].to_numpy(),
            "data": historico["data"].to_numpy(),
            "previsto": modelo.prever(historico),
        })
        alinhado = alvo[["ticker", "data"]].merge(previsto, on=["ticker", "data"], how="left")
        return alinhado["previsto"].to_numpy(dtype=object)

    def _validacao_cruzada(self, modelo: ModeloClassificacao, dados: pd.DataFrame) -> dict:
        """F1 macro e MCC em cada fold da validação temporal expurgada."""
        dados = dados.sort_values(["data", "ticker"]).reset_index(drop=True)
        f1s, mccs = [], []
        for idx_treino, idx_val in self.divisor.split(dados, groups=dados["data"]):
            copia = copy.deepcopy(modelo)
            copia.ajustar(dados.iloc[idx_treino])
            validacao = dados.iloc[idx_val]
            previsto = self._prever_alinhado(copia, validacao)
            validos = pd.notna(previsto)
            f1s.append(float(f1_score(validacao["alvo"][validos], previsto[validos],
                                      labels=list(CLASSES), average="macro", zero_division=0)))
            mccs.append(float(matthews_corrcoef(validacao["alvo"][validos], previsto[validos])))
        return {
            "f1_media": float(np.mean(f1s)), "f1_desvio": float(np.std(f1s)),
            "mcc_media": float(np.mean(mccs)), "mcc_desvio": float(np.std(mccs)),
            "f1_por_fold": f1s, "mcc_por_fold": mccs,
        }

    def _validacao_cruzada_aleatoria(self, dados: pd.DataFrame) -> float:
        """F1 macro médio de um classificador aleatório estratificado (referência)."""
        dados = dados.sort_values(["data", "ticker"]).reset_index(drop=True)
        f1s = []
        for idx_treino, idx_val in self.divisor.split(dados, groups=dados["data"]):
            aleatorio = DummyClassifier(strategy="stratified", random_state=self.config.semente)
            aleatorio.fit(np.zeros((len(idx_treino), 1)), dados["alvo"].iloc[idx_treino])
            previsto = aleatorio.predict(np.zeros((len(idx_val), 1)))
            f1s.append(float(f1_score(dados["alvo"].iloc[idx_val], previsto,
                                      labels=list(CLASSES), average="macro", zero_division=0)))
        return float(np.mean(f1s))

    def _rf_experimental(self, features: list[str]) -> ModeloRandomForest:
        modelo = ModeloRandomForest(features, self.config.semente, self.config.n_jobs)
        modelo.hiperparametros = dict(PARAMETROS_EXPERIMENTO_RF)
        return modelo

    # ══════════════════════════════════════════════════════════════════════
    # Experimentos
    # ══════════════════════════════════════════════════════════════════════
    def _experimento_limiar(self, treino: pd.DataFrame) -> tuple[list[dict], RegraLimiar]:
        """
        Reavalia o limiar da variável-alvo com as variáveis originais.

        Como a proporção das classes muda com o limiar, a escolha usa o MCC,
        que desconta o acerto esperado ao acaso; o F1 macro de um
        classificador aleatório também é registrado para comparação.
        """
        resultados, regras = [], []
        for tipo, valor in self.config.limiares_avaliados:
            limiar = RegraLimiar(tipo, valor)
            regras.append(limiar)
            dados = self._rotular_novamente(treino, limiar)
            cv = self._validacao_cruzada(self._rf_experimental(CalculadoraIndicadores.FEATURES_ORIGINAIS), dados)
            resultados.append({
                "limiar": limiar.como_dict(),
                "limiar_medio": float(np.nanmean(dados["limiar_alvo"])),
                "distribuicao": Rotulador.distribuicao(dados["alvo"]),
                "f1_cv_media": cv["f1_media"], "f1_cv_desvio": cv["f1_desvio"],
                "mcc_cv_media": cv["mcc_media"], "mcc_cv_desvio": cv["mcc_desvio"],
                "f1_aleatorio": self._validacao_cruzada_aleatoria(dados),
            })
            log.info("  limiar %s: F1 %.4f | MCC %.4f", limiar.descricao(),
                     cv["f1_media"], cv["mcc_media"])
        melhor = int(np.argmax([r["mcc_cv_media"] for r in resultados]))
        escolhido = regras[melhor]
        log.info("  limiar escolhido: %s", escolhido.descricao())
        return resultados, escolhido

    def _experimento_features(self, treino: pd.DataFrame) -> tuple[list[dict], list[str]]:
        """Compara as variáveis originais, as novas e o conjunto completo."""
        conjuntos = {
            "originais": CalculadoraIndicadores.FEATURES_ORIGINAIS,
            "novas": CalculadoraIndicadores.FEATURES_NOVAS,
            "completas": CalculadoraIndicadores.FEATURES_COMPLETAS,
        }
        resultados = []
        for nome, features in conjuntos.items():
            cv = self._validacao_cruzada(self._rf_experimental(features), treino)
            resultados.append({
                "conjunto": nome, "n_features": len(features),
                "f1_cv_media": cv["f1_media"], "f1_cv_desvio": cv["f1_desvio"],
                "mcc_cv_media": cv["mcc_media"],
            })
            log.info("  variáveis %s (%d): F1 %.4f | MCC %.4f", nome, len(features),
                     cv["f1_media"], cv["mcc_media"])
        melhor = max(resultados, key=lambda r: r["f1_cv_media"])["conjunto"]
        log.info("  conjunto escolhido: %s", melhor)
        return resultados, list(conjuntos[melhor])

    def _reproduzir_tc1(self, base: pd.DataFrame) -> dict:
        """Configuração exata do TC I (8 variáveis, limiar de 1,5%) nos dados reais."""
        conjuntos = self._montar_conjuntos(base, RegraLimiar("fixo", 0.015))
        modelo = ModeloRandomForestTC1(CalculadoraIndicadores.FEATURES_ORIGINAIS,
                                       self.config.semente, self.config.n_jobs)
        cv = self._validacao_cruzada(modelo, conjuntos["treino"])
        modelo.ajustar(conjuntos["treino"])
        resultado = {
            "limiar": RegraLimiar("fixo", 0.015).como_dict(),
            "cv": cv,
            "teste": self.avaliador.avaliar(conjuntos["teste"]["alvo"], modelo.prever(conjuntos["teste"])),
            "oos": self.avaliador.avaliar(conjuntos["oos"]["alvo"], modelo.prever(conjuntos["oos"])),
            "importancias": modelo.importancias(),
        }
        log.info("  referência TC I: F1 teste %.4f | F1 OOS %.4f",
                 resultado["teste"]["f1_macro"], resultado["oos"]["f1_macro"])
        return resultado

    # ══════════════════════════════════════════════════════════════════════
    # Treinamento dos modelos
    # ══════════════════════════════════════════════════════════════════════
    def _treinar_modelos(self, treino: pd.DataFrame, features: list[str]) -> dict:
        cfg = self.config
        modelos: dict[str, ModeloClassificacao] = {
            "random_forest": ModeloRandomForest(features, cfg.semente, cfg.n_jobs),
            "svm": ModeloSVM(features, cfg.semente, cfg.n_jobs),
            "mlp": ModeloMLP(features, cfg.semente, cfg.n_jobs),
        }
        for modelo in modelos.values():
            inicio = time.perf_counter()
            modelo.buscar_hiperparametros(treino, self.divisor, reduzida=self.modo_rapido)
            modelo.ajustar(treino)
            log.info("  %s: F1 CV %.4f (%.0f s) %s", modelo.nome,
                     modelo.resultado_busca["f1_cv_media"], time.perf_counter() - inicio,
                     modelo.descricao_hiperparametros())
        modelos["lstm"] = self._treinar_lstm(treino)
        return modelos

    def _novo_previsor(self, **ajustes) -> PrevisorLSTM:
        cfg = self.config
        parametros = {"horizontes": cfg.horizonte_maximo_previsao, "semente": cfg.semente}
        if self.modo_rapido:
            parametros.update(epocas=3, paciencia=2)
        parametros.update(ajustes)
        return PrevisorLSTM(**parametros)

    def _treinar_lstm(self, treino: pd.DataFrame) -> ModeloLSTMClassificacao:
        """Escolhe a configuração da LSTM pela perda em validação e treina."""
        inicio = time.perf_counter()
        grade = GRADE_LSTM[:1] if self.modo_rapido else GRADE_LSTM
        avaliacoes = []
        for ajustes in grade:
            previsor = self._novo_previsor(**ajustes).ajustar(treino)
            avaliacoes.append({**ajustes, "perda_validacao": previsor.perda_validacao,
                               "epocas_treinadas": previsor.epocas_treinadas})
            log.info("  LSTM %s: perda de validação %.4f", ajustes, previsor.perda_validacao)
        melhor = min(avaliacoes, key=lambda a: a["perda_validacao"])
        ajustes = {"janela": melhor["janela"], "oculto": melhor["oculto"]}

        modelo = ModeloLSTMClassificacao(self._novo_previsor(**ajustes), self.config.horizonte,
                                         self.config.semente)
        cv = self._validacao_cruzada(modelo, treino)
        modelo.ajustar(treino)
        modelo.resultado_busca = {
            "melhores_hiperparametros": ajustes,
            "f1_cv_media": cv["f1_media"], "f1_cv_desvio": cv["f1_desvio"],
            "f1_cv_por_fold": cv["f1_por_fold"], "combinacoes_testadas": len(grade),
            "grade": avaliacoes,
        }
        log.info("  LSTM: F1 CV %.4f (%.0f s) %s", cv["f1_media"], time.perf_counter() - inicio, ajustes)
        return modelo

    def _treinar_producao(self, modelos: dict, estudo: pd.DataFrame) -> dict:
        """Retreina cada modelo, com os mesmos hiperparâmetros, em todo o estudo."""
        base_modelos = estudo[estudo["ticker"].isin(self.config.tickers_treino)]
        producao = {}
        for chave, modelo in modelos.items():
            if isinstance(modelo, ModeloLSTMClassificacao):
                copia = ModeloLSTMClassificacao(
                    self._novo_previsor(janela=modelo.previsor.janela, oculto=modelo.previsor.oculto),
                    self.config.horizonte, self.config.semente)
            else:
                copia = copy.deepcopy(modelo)
                copia.estimador = None
            copia.resultado_busca = modelo.resultado_busca
            copia.ajustar(base_modelos)
            producao[chave] = copia
        return producao

    # ══════════════════════════════════════════════════════════════════════
    # Avaliação
    # ══════════════════════════════════════════════════════════════════════
    def _avaliar_conjunto(self, modelo, dados: pd.DataFrame, com_intervalo: bool = False) -> dict:
        previsto = self._prever_alinhado(modelo, dados)
        validos = pd.notna(previsto)
        resultado = self.avaliador.avaliar(dados["alvo"][validos], previsto[validos])
        if com_intervalo:
            n = 200 if self.modo_rapido else 1000
            intervalos = self.avaliador.intervalo_confianca(
                dados["alvo"][validos], previsto[validos], dados["data"][validos], n_reamostragens=n)
            resultado["f1_ic95"] = intervalos["f1_macro"]
            resultado["mcc_ic95"] = intervalos["mcc"]
        return resultado

    def _avaliar_modelos(self, modelos: dict, conjuntos: dict) -> dict:
        avaliacao = {}
        for chave, modelo in modelos.items():
            busca = modelo.resultado_busca or {}
            avaliacao[chave] = {
                "nome": modelo.nome,
                "hiperparametros": modelo.descricao_hiperparametros(),
                "cv": {
                    "f1_media": busca.get("f1_cv_media"),
                    "f1_desvio": busca.get("f1_cv_desvio"),
                    "f1_por_fold": busca.get("f1_cv_por_fold"),
                    "combinacoes_testadas": busca.get("combinacoes_testadas"),
                },
                "teste": self._avaliar_conjunto(modelo, conjuntos["teste"], com_intervalo=True),
                "oos": self._avaliar_conjunto(modelo, conjuntos["oos"], com_intervalo=True),
                "oos_periodo_completo": self._avaliar_conjunto(modelo, conjuntos["oos_completo"]),
                "importancias": modelo.importancias(),
                "tempo_treino_s": round(modelo.tempo_treino_s or 0.0, 2),
            }
            if chave == "lstm":
                avaliacao[chave]["grade_lstm"] = busca.get("grade")
            t = avaliacao[chave]["teste"]
            log.info("  %s: teste F1 %.4f [%.4f; %.4f] MCC %.4f | OOS F1 %.4f", modelo.nome,
                     t["f1_macro"], *t["f1_ic95"], t["mcc"], avaliacao[chave]["oos"]["f1_macro"])
        return avaliacao

    def _avaliar_prospectivo(self, producao: dict, prospectivo: pd.DataFrame) -> dict:
        resultado = {}
        for chave, modelo in producao.items():
            geral = self._avaliar_conjunto(modelo, prospectivo, com_intervalo=True)
            por_ativo = {}
            for ticker, grupo in prospectivo.groupby("ticker"):
                por_ativo[ticker] = self._avaliar_conjunto(modelo, grupo)["f1_macro"]
            resultado[chave] = {**geral, "f1_por_ativo": por_ativo}
            log.info("  %s (prospectivo): F1 %.4f MCC %.4f", modelo.nome, geral["f1_macro"], geral["mcc"])
        return resultado

    def _avaliar_referencias(self, treino: pd.DataFrame, conjuntos: dict) -> dict:
        """Classificadores ingênuos: aleatório estratificado e classe majoritária."""
        referencias = {}
        for estrategia, nome in (("stratified", "Aleatório estratificado"),
                                 ("most_frequent", "Classe majoritária")):
            modelo = DummyClassifier(strategy=estrategia, random_state=self.config.semente)
            modelo.fit(np.zeros((len(treino), 1)), treino["alvo"])
            referencias[estrategia] = {"nome": nome}
            for conjunto in ("teste", "oos", "prospectivo"):
                dados = conjuntos[conjunto]
                previsto = modelo.predict(np.zeros((len(dados), 1)))
                referencias[estrategia][conjunto] = self.avaliador.avaliar(dados["alvo"], previsto)
        return referencias

    def _retornos_reais(self, dados: pd.DataFrame, limite: pd.Timestamp | None) -> pd.DataFrame:
        """Retorno logarítmico observado de 1 a H pregões à frente (por ativo)."""
        base = self._base if limite is None else self._base[self._base["data"] <= limite]
        base = base.sort_values(["ticker", "data"])
        tabela = base[["ticker", "data", "close"]].copy()
        log_preco = np.log(tabela["close"])
        for h in range(1, self.config.horizonte_maximo_previsao + 1):
            tabela[f"real_{h}"] = log_preco.groupby(tabela["ticker"]).shift(-h) - log_preco
        return dados[["ticker", "data"]].merge(tabela, on=["ticker", "data"], how="left")

    def _previsoes_lstm(self, previsor: PrevisorLSTM, dados: pd.DataFrame) -> pd.DataFrame:
        historico = self._base[
            self._base["ticker"].isin(dados["ticker"].unique()) & (self._base["data"] <= dados["data"].max())
        ]
        previsto = previsor.prever(historico).drop(columns=["close"])
        return dados[["ticker", "data"]].merge(previsto, on=["ticker", "data"], how="left")

    def _metricas_regressao(self, previsto: pd.DataFrame, real: pd.DataFrame) -> dict:
        avaliador = AvaliadorRegressao()
        resultado = {}
        for h in HORIZONTES_RELATORIO:
            mascara = previsto[f"r_{h}"].notna() & real[f"real_{h}"].notna()
            base = real.loc[mascara, "close"].to_numpy()
            preco_real = base * np.exp(real.loc[mascara, f"real_{h}"].to_numpy())
            preco_lstm = base * np.exp(previsto.loc[mascara, f"r_{h}"].to_numpy())
            lstm = avaliador.avaliar(preco_real, preco_lstm, base)
            ingenuo = avaliador.avaliar(preco_real, base, base)
            resultado[str(h)] = {
                "lstm": lstm,
                "ingenuo": {k: ingenuo[k] for k in ("rmse", "mae", "mape", "n_amostras")},
                "ganho_rmse_vs_ingenuo": 1 - lstm["rmse"] / ingenuo["rmse"],
            }
        return resultado

    def _avaliar_regressao(self, previsor: PrevisorLSTM, conjuntos: dict) -> dict:
        """RMSE, MAE, MAPE e direção da LSTM versus o passeio aleatório."""
        fim_estudo = pd.Timestamp(self.config.fim_estudo)
        resultado = {}
        for conjunto in ("teste", "oos"):
            dados = conjuntos[conjunto]
            previsto = self._previsoes_lstm(previsor, dados)
            real = self._retornos_reais(dados, fim_estudo)
            resultado[conjunto] = self._metricas_regressao(previsto, real)
            if conjunto == "teste":
                faixas = self._faixas_confianca(previsto, real, dados["volatilidade_20"].to_numpy())
        for h in HORIZONTES_RELATORIO:
            r = resultado["teste"][str(h)]
            log.info("  LSTM h=%d: RMSE R$ %.3f vs ingênuo R$ %.3f | direção %.1f%%", h,
                     r["lstm"]["rmse"], r["ingenuo"]["rmse"], 100 * r["lstm"]["acerto_direcao"])
        resultado["faixas"] = faixas
        return resultado

    def _faixas_confianca(self, previsto: pd.DataFrame, real: pd.DataFrame,
                          volatilidade: np.ndarray) -> dict:
        """
        Quantis empíricos (10% e 90%) dos erros da LSTM no teste, por horizonte.

        São calculadas duas versões da faixa de 80%, sem supor normalidade:
          * fixa (q10, q90): quantis do erro em retorno;
          * ajustada pela volatilidade (z10, z90): quantis do erro dividido
            pela volatilidade de 20 pregões x raiz(h). No uso, os quantis são
            multiplicados pela volatilidade atual, de modo que a faixa se
            alarga em períodos turbulentos e se estreita em períodos calmos.
        """
        faixas = {}
        for h in range(1, self.config.horizonte_maximo_previsao + 1):
            erro = real[f"real_{h}"] - previsto[f"r_{h}"]
            padronizado = (erro / (volatilidade * np.sqrt(h))).dropna()
            erro = erro.dropna()
            faixas[str(h)] = {
                "q10": float(erro.quantile(0.10)),
                "q50": float(erro.quantile(0.50)),
                "q90": float(erro.quantile(0.90)),
                "z10": float(padronizado.quantile(0.10)),
                "z90": float(padronizado.quantile(0.90)),
            }
        return faixas

    def _avaliar_regressao_prospectiva(self, previsor: PrevisorLSTM, prospectivo: pd.DataFrame,
                                       faixas: dict) -> dict:
        previsto = self._previsoes_lstm(previsor, prospectivo)
        real = self._retornos_reais(prospectivo, None)
        resultado = self._metricas_regressao(previsto, real)
        volatilidade = prospectivo["volatilidade_20"].to_numpy()
        for h in HORIZONTES_RELATORIO:
            erro = real[f"real_{h}"] - previsto[f"r_{h}"]
            validos = erro.notna()
            faixa = faixas[str(h)]
            escala = volatilidade * np.sqrt(h)
            dentro_fixa = (erro >= faixa["q10"]) & (erro <= faixa["q90"])
            dentro_ajustada = (erro >= faixa["z10"] * escala) & (erro <= faixa["z90"] * escala)
            resultado[str(h)]["cobertura_faixa_fixa"] = float(dentro_fixa[validos].mean())
            resultado[str(h)]["cobertura_faixa_volatilidade"] = float(dentro_ajustada[validos].mean())
            log.info("  LSTM h=%d (prospectivo): cobertura da faixa de 80%%: fixa %.1f%% | ajustada %.1f%%",
                     h, 100 * resultado[str(h)]["cobertura_faixa_fixa"],
                     100 * resultado[str(h)]["cobertura_faixa_volatilidade"])
        return resultado

    def _backtest(self, modelos: dict, dados: pd.DataFrame) -> dict:
        backtester = Backtester()
        resultado, curvas = {}, {}
        for chave, modelo in modelos.items():
            sinais = dados[["data", "ticker", "close"]].assign(sinal=self._prever_alinhado(modelo, dados))
            simulacao = backtester.simular(sinais)
            resultado[chave] = simulacao.estrategia
            curvas[chave] = simulacao.curva["estrategia"].round(4).tolist()
        resultado["comprar_e_manter"] = simulacao.referencia
        curvas["comprar_e_manter"] = simulacao.curva["referencia"].round(4).tolist()
        curvas["datas"] = simulacao.curva["data"].dt.strftime("%Y-%m-%d").tolist()
        resultado["curvas"] = curvas
        return resultado

    # ══════════════════════════════════════════════════════════════════════
    # Persistência
    # ══════════════════════════════════════════════════════════════════════
    def _informacoes(self, conjuntos, limiar, features, limpeza, principal) -> dict:
        import sklearn

        try:
            import torch
            versao_torch = torch.__version__
        except ImportError:  # pragma: no cover
            versao_torch = None
        cfg = self.config
        estudo, prospectivo = conjuntos["estudo"], conjuntos["prospectivo"]
        return {
            "data_treinamento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fonte_dados": "Yahoo Finance (yfinance, preços ajustados por proventos)",
            "periodo_estudo": [f"{estudo['data'].min():%Y-%m-%d}", f"{estudo['data'].max():%Y-%m-%d}"],
            "periodo_prospectivo": [f"{prospectivo['data'].min():%Y-%m-%d}",
                                    f"{prospectivo['data'].max():%Y-%m-%d}"],
            "inicio_teste": f"{conjuntos['inicio_teste']:%Y-%m-%d}",
            "tickers_treino": list(cfg.tickers_treino),
            "ticker_out_of_sample": cfg.ticker_oos,
            "horizonte": cfg.horizonte,
            "limiar": limiar.como_dict(),
            "features": features,
            "n_folds": cfg.n_folds,
            "registros": {
                "estudo_rotulado": int(len(estudo)),
                "treino": int(len(conjuntos["treino"])),
                "teste": int(len(conjuntos["teste"])),
                "oos": int(len(conjuntos["oos"])),
                "oos_periodo_completo": int(len(conjuntos["oos_completo"])),
                "prospectivo": int(len(prospectivo)),
            },
            "distribuicao_classes": {
                nome: Rotulador.distribuicao(conjuntos[nome]["alvo"])
                for nome in ("treino", "teste", "oos", "prospectivo")
            },
            "limpeza": limpeza,
            "modelo_principal": principal,
            "versoes": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "torch": versao_torch,
            },
        }

    def _salvar(self, producao: dict, conjuntos: dict, modelos_teste: dict) -> None:
        pasta = self.caminhos.modelos
        pasta.mkdir(parents=True, exist_ok=True)
        for modelo in producao.values():
            modelo.salvar(pasta)
        with open(self.caminhos.metricas, "w", encoding="utf-8") as arquivo:
            json.dump(_limpar_json(self.metricas), arquivo, indent=2, ensure_ascii=False)

        # Base com indicadores e rótulos (limiar escolhido) de todas as datas.
        rotulada = Rotulador(self.config.horizonte, conjuntos["limiar"]).adicionar_alvo(self._base)
        self.repositorio.salvar_features(rotulada)

        # Previsões nos conjuntos de avaliação, usadas na tela de desempenho.
        partes = []
        for nome, modelos in (("teste", modelos_teste), ("oos", modelos_teste), ("prospectivo", producao)):
            dados = conjuntos[nome]
            tabela = dados[["data", "ticker", "close", "retorno_futuro", "alvo"]].copy()
            tabela["conjunto"] = nome
            for chave, modelo in modelos.items():
                tabela[f"pred_{chave}"] = self._prever_alinhado(modelo, dados)
            previsto = self._previsoes_lstm(modelos["lstm"].previsor, dados)
            tabela["lstm_r_5"] = previsto["r_5"].to_numpy()
            partes.append(tabela)
        previsoes = pd.concat(partes, ignore_index=True)
        self.repositorio.salvar_previsoes(previsoes)


def _limpar_json(objeto):
    """Converte tipos NumPy/pandas e substitui NaN por None para gravar em JSON."""
    if isinstance(objeto, dict):
        return {str(k): _limpar_json(v) for k, v in objeto.items()}
    if isinstance(objeto, (list, tuple)):
        return [_limpar_json(v) for v in objeto]
    if isinstance(objeto, (np.floating, float)):
        valor = float(objeto)
        return None if np.isnan(valor) or np.isinf(valor) else round(valor, 6)
    if isinstance(objeto, np.integer):
        return int(objeto)
    if isinstance(objeto, pd.Timestamp):
        return objeto.strftime("%Y-%m-%d")
    return objeto
