"""
Classe-base dos classificadores.

Define a interface comum (ajustar, prever, prever_proba, salvar, carregar) e
implementa a busca de hiperparâmetros em grade com a validação temporal
expurgada. As subclasses só precisam dizer qual estimador construir e qual
grade de hiperparâmetros testar.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from econom_ia.config import CLASSES


class ModeloClassificacao(ABC):
    """Interface comum dos modelos que classificam em compra, venda ou neutro."""

    chave: str = ""   # identificador usado em arquivos e no metricas.json
    nome: str = ""    # nome exibido na interface e no artigo

    def __init__(self, features: list[str], semente: int = 42, n_jobs: int = 1) -> None:
        self.features = list(features)
        self.semente = semente
        self.n_jobs = n_jobs
        self.hiperparametros: dict = {}
        self.estimador = None
        self.resultado_busca: dict | None = None
        self.tempo_treino_s: float | None = None

    # ── Pontos de extensão das subclasses ──────────────────────────────────
    @abstractmethod
    def _construir(self, para_busca: bool = False):
        """Cria o estimador (ainda não treinado) com os hiperparâmetros padrão."""

    @abstractmethod
    def grade_busca(self) -> dict:
        """Grade de hiperparâmetros avaliada na busca (nomes do scikit-learn)."""

    def importancias(self) -> dict[str, float] | None:
        """Importância global de cada variável, quando o modelo a fornece."""
        return None

    # ── Treinamento ────────────────────────────────────────────────────────
    def _matrizes(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        dados = df.sort_values(["data", "ticker"])
        return dados[self.features].to_numpy(dtype=float), dados["alvo"].to_numpy(dtype=object)

    def buscar_hiperparametros(self, df: pd.DataFrame, divisor, reduzida: bool = False) -> dict:
        """
        Busca em grade com validação cruzada temporal expurgada.

        A métrica otimizada é o F1-Score macro. Retorna o melhor conjunto de
        hiperparâmetros e guarda o resumo da busca em "resultado_busca".

        Args:
            reduzida: testa só o primeiro valor de cada hiperparâmetro
                (usado nos testes automatizados, para rodar rápido).
        """
        dados = df.sort_values(["data", "ticker"])
        X, y = self._matrizes(dados)
        grade = self.grade_busca()
        if reduzida:
            grade = {nome: valores[:1] for nome, valores in grade.items()}
        busca = GridSearchCV(
            self._construir(para_busca=True),
            grade,
            scoring="f1_macro",
            cv=divisor,
            n_jobs=self.n_jobs,
            refit=False,
            error_score="raise",
        )
        busca.fit(X, y, groups=dados["data"].to_numpy())
        melhor = int(busca.best_index_)
        self.hiperparametros = dict(busca.best_params_)
        self.resultado_busca = {
            "melhores_hiperparametros": _serializavel(busca.best_params_),
            "f1_cv_media": float(busca.cv_results_["mean_test_score"][melhor]),
            "f1_cv_desvio": float(busca.cv_results_["std_test_score"][melhor]),
            "f1_cv_por_fold": [
                float(busca.cv_results_[f"split{k}_test_score"][melhor])
                for k in range(divisor.get_n_splits())
            ],
            "combinacoes_testadas": int(len(busca.cv_results_["params"])),
        }
        return self.hiperparametros

    def ajustar(self, df: pd.DataFrame) -> ModeloClassificacao:
        """Treina o modelo final com os hiperparâmetros atuais."""
        X, y = self._matrizes(df)
        estimador = self._construir()
        if self.hiperparametros:
            estimador.set_params(**self.hiperparametros)
        inicio = time.perf_counter()
        estimador.fit(X, y)
        self.tempo_treino_s = time.perf_counter() - inicio
        self.estimador = estimador
        return self

    # ── Inferência ─────────────────────────────────────────────────────────
    def _verificar_treino(self) -> None:
        if self.estimador is None:
            raise RuntimeError(f"O modelo {self.nome} ainda não foi treinado.")

    def prever(self, df: pd.DataFrame) -> np.ndarray:
        """Classe prevista para cada linha (na mesma ordem de df)."""
        self._verificar_treino()
        return self.estimador.predict(df[self.features].to_numpy(dtype=float)).astype(object)

    def prever_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Probabilidade de cada classe (colunas venda, neutro, compra)."""
        self._verificar_treino()
        proba = self.estimador.predict_proba(df[self.features].to_numpy(dtype=float))
        tabela = pd.DataFrame(proba, columns=list(self.estimador.classes_), index=df.index)
        return tabela.reindex(columns=list(CLASSES), fill_value=0.0)

    # ── Persistência ───────────────────────────────────────────────────────
    def caminho_arquivo(self, pasta: Path) -> Path:
        return Path(pasta) / f"{self.chave}.pkl"

    def salvar(self, pasta: Path) -> Path:
        """Serializa o objeto completo (modelo e metadados) com joblib."""
        caminho = self.caminho_arquivo(pasta)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, caminho, compress=3)
        return caminho

    def descricao_hiperparametros(self) -> dict:
        return _serializavel(self.hiperparametros)


def carregar_modelo(caminho: Path) -> ModeloClassificacao:
    """Carrega um modelo salvo por ModeloClassificacao.salvar()."""
    modelo = joblib.load(caminho)
    if not isinstance(modelo, ModeloClassificacao):
        raise TypeError(f"O arquivo {caminho} não contém um modelo do Econom-IA.")
    return modelo


def _serializavel(parametros: dict) -> dict:
    """Converte valores (tuplas, tipos numpy) para gravação em JSON."""
    saida = {}
    for chave, valor in parametros.items():
        nome = chave.split("__")[-1]
        if isinstance(valor, tuple):
            valor = list(valor)
        elif isinstance(valor, np.generic):
            valor = valor.item()
        saida[nome] = valor
    return saida
