"""
Métricas de avaliação dos modelos.

Classificação: Precision, Recall e F1-Score em versão macro (média simples
entre as três classes), acurácia e o coeficiente de correlação de Matthews
(MCC), que vale 0 para um classificador equivalente ao acaso e 1 para um
classificador perfeito, independentemente do balanceamento das classes.

Predição de preço: RMSE, MAE, MAPE e taxa de acerto da direção do movimento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from econom_ia.config import CLASSES


class AvaliadorClassificacao:
    """Calcula as métricas de classificação e seus intervalos de confiança."""

    classes = list(CLASSES)

    def avaliar(self, y_real, y_previsto) -> dict:
        y_real = np.asarray(y_real, dtype=object)
        y_previsto = np.asarray(y_previsto, dtype=object)
        return {
            "f1_macro": self._f1(y_real, y_previsto),
            "precision_macro": float(precision_score(
                y_real, y_previsto, labels=self.classes, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(
                y_real, y_previsto, labels=self.classes, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_real, y_previsto)),
            "mcc": float(matthews_corrcoef(y_real, y_previsto)),
            "f1_por_classe": {
                classe: float(f1_score(y_real, y_previsto, labels=[classe], average="macro", zero_division=0))
                for classe in self.classes
            },
            "matriz_confusao": confusion_matrix(y_real, y_previsto, labels=self.classes).tolist(),
            "n_amostras": int(len(y_real)),
        }

    def _f1(self, y_real, y_previsto) -> float:
        return float(f1_score(y_real, y_previsto, labels=self.classes, average="macro", zero_division=0))

    def intervalo_confianca(
        self,
        y_real,
        y_previsto,
        datas,
        n_reamostragens: int = 1000,
        tamanho_bloco: int = 20,
        confianca: float = 0.95,
        semente: int = 42,
    ) -> dict[str, list[float]]:
        """
        Intervalos de confiança do F1 macro e do MCC por bootstrap em blocos.

        Reamostrar blocos de pregões consecutivos (em vez de linhas isoladas)
        preserva a dependência temporal e a correlação entre as ações de um
        mesmo dia, evitando intervalos artificialmente estreitos. Se o
        intervalo do MCC não contém o zero, o desempenho é superior ao acaso.
        """
        y_real = np.asarray(y_real, dtype=object)
        y_previsto = np.asarray(y_previsto, dtype=object)
        datas = pd.to_datetime(np.asarray(datas)).to_numpy()
        unicas = np.unique(datas)
        posicao = {d: i for i, d in enumerate(unicas)}
        indice_data = np.array([posicao[d] for d in datas])
        linhas_por_data = [np.flatnonzero(indice_data == i) for i in range(len(unicas))]

        gerador = np.random.default_rng(semente)
        n_blocos = int(np.ceil(len(unicas) / tamanho_bloco))
        f1s = np.empty(n_reamostragens)
        mccs = np.empty(n_reamostragens)
        for r in range(n_reamostragens):
            inicios = gerador.integers(0, max(len(unicas) - tamanho_bloco, 1), size=n_blocos)
            dias = np.concatenate([np.arange(i, min(i + tamanho_bloco, len(unicas))) for i in inicios])
            linhas = np.concatenate([linhas_por_data[d] for d in dias])
            f1s[r] = self._f1(y_real[linhas], y_previsto[linhas])
            mccs[r] = matthews_corrcoef(y_real[linhas], y_previsto[linhas])
        alfa = (1 - confianca) / 2
        return {
            "f1_macro": [float(np.quantile(f1s, alfa)), float(np.quantile(f1s, 1 - alfa))],
            "mcc": [float(np.quantile(mccs, alfa)), float(np.quantile(mccs, 1 - alfa))],
        }


class AvaliadorRegressao:
    """Métricas da previsão de preço, comparáveis a um modelo ingênuo."""

    @staticmethod
    def avaliar(preco_real, preco_previsto, preco_base) -> dict:
        """
        Args:
            preco_real: preço observado no horizonte avaliado.
            preco_previsto: preço estimado pelo modelo.
            preco_base: preço no momento da previsão (último preço conhecido),
                usado para medir o acerto da direção do movimento.
        """
        real = np.asarray(preco_real, dtype=float)
        previsto = np.asarray(preco_previsto, dtype=float)
        base = np.asarray(preco_base, dtype=float)
        erro = previsto - real
        movimento_real = np.sign(real - base)
        movimento_previsto = np.sign(previsto - base)
        validos = movimento_real != 0
        return {
            "rmse": float(np.sqrt(np.mean(erro ** 2))),
            "mae": float(np.mean(np.abs(erro))),
            "mape": float(np.mean(np.abs(erro / real)) * 100),
            "acerto_direcao": float(np.mean(movimento_previsto[validos] == movimento_real[validos])),
            "n_amostras": int(len(real)),
        }
