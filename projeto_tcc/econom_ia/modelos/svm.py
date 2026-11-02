"""
SVM (Support Vector Machine) com kernel RBF.

As variáveis são padronizadas antes do SVM, pois o algoritmo é sensível à
escala. O SVM não produz probabilidades diretamente; a "confiança" exibida na
interface é a normalização (softmax) das pontuações de decisão de cada classe,
o que mantém a classe prevista coerente com a regra de decisão do modelo.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from econom_ia.modelos.base import ModeloClassificacao


class SVCComPontuacao(SVC):
    """SVC cuja predict_proba é o softmax das pontuações de decisão (um-contra-todos)."""

    def predict_proba(self, X):  # noqa: N803 (convenção do scikit-learn)
        pontuacao = self.decision_function(X)
        exponencial = np.exp(pontuacao - pontuacao.max(axis=1, keepdims=True))
        return exponencial / exponencial.sum(axis=1, keepdims=True)


class ModeloSVM(ModeloClassificacao):
    chave = "svm"
    nome = "SVM"

    def _construir(self, para_busca: bool = False):
        return Pipeline([
            ("padronizacao", StandardScaler()),
            ("modelo", SVCComPontuacao(
                kernel="rbf",
                class_weight="balanced",
                decision_function_shape="ovr",
                # Empates na votação são decididos pela maior pontuação, o que
                # deixa a classe prevista igual à de maior "confiança".
                break_ties=True,
                random_state=self.semente,
            )),
        ])

    def grade_busca(self) -> dict:
        return {
            "modelo__C": [0.1, 0.3, 1.0, 3.0, 10.0],
            "modelo__gamma": ["scale", 0.01, 0.03, 0.1],
        }
