"""
Modelos de Machine Learning do sistema.

Todos os classificadores herdam de ModeloClassificacao e, por isso, são
treinados, avaliados, salvos e usados pela interface da mesma forma
(polimorfismo). A LSTM também é usada como previsor de preço.
"""

from econom_ia.modelos.base import ModeloClassificacao, carregar_modelo
from econom_ia.modelos.lstm import ModeloLSTMClassificacao, PrevisorLSTM
from econom_ia.modelos.mlp import ModeloMLP
from econom_ia.modelos.random_forest import ModeloRandomForest
from econom_ia.modelos.svm import ModeloSVM

__all__ = [
    "ModeloClassificacao",
    "ModeloLSTMClassificacao",
    "ModeloMLP",
    "ModeloRandomForest",
    "ModeloSVM",
    "PrevisorLSTM",
    "carregar_modelo",
]
