"""
Divisões temporais sem vazamento de informação.

Como a variável-alvo de um pregão usa o preço de "horizonte" pregões à
frente, as amostras de treino que ficam imediatamente antes do período de
validação carregam no rótulo informação desse período. Por isso essas
amostras são expurgadas (técnica de purging descrita por López de Prado).
As divisões são feitas por data, e não por linha, porque a base empilha
vários ativos no mesmo pregão.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class DivisorTemporalPurgado:
    """
    Validação cruzada em janela expansiva (walk-forward) com expurgo.

    Compatível com o scikit-learn (GridSearchCV, cross_validate): as datas de
    cada amostra devem ser informadas no argumento "groups".
    """

    def __init__(self, n_folds: int = 5, horizonte: int = 5) -> None:
        self.n_folds = n_folds
        self.horizonte = horizonte

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: N803 (API do sklearn)
        return self.n_folds

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:  # noqa: N803
        if groups is None:
            raise ValueError("Informe as datas das amostras no parâmetro 'groups'.")
        datas = pd.to_datetime(np.asarray(groups)).to_numpy()
        unicas = np.unique(datas)
        tamanho_bloco = len(unicas) // (self.n_folds + 1)
        if tamanho_bloco <= self.horizonte:
            raise ValueError("Poucas datas para o número de folds solicitado.")

        for k in range(self.n_folds):
            inicio_val = (k + 1) * tamanho_bloco
            fim_val = inicio_val + tamanho_bloco if k < self.n_folds - 1 else len(unicas)
            # As "horizonte" datas anteriores à validação são expurgadas do treino.
            limite_treino = unicas[inicio_val - self.horizonte]
            idx_treino = np.flatnonzero(datas < limite_treino)
            idx_val = np.flatnonzero((datas >= unicas[inicio_val]) & (datas <= unicas[fim_val - 1]))
            yield idx_treino, idx_val


def dividir_por_data(
    df: pd.DataFrame, proporcao_treino: float = 0.7, horizonte: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Divide a base em treino e teste respeitando a ordem cronológica.

    Os primeiros "proporcao_treino" das datas formam o treino e o restante o
    teste. As "horizonte" datas imediatamente anteriores ao teste são
    descartadas do treino (expurgo), pois seus rótulos dependem de preços que
    já pertencem ao período de teste.

    Returns:
        (treino, teste, primeira data do teste)
    """
    datas = np.sort(df["data"].unique())
    indice_corte = int(len(datas) * proporcao_treino)
    inicio_teste = pd.Timestamp(datas[indice_corte])
    limite_treino = pd.Timestamp(datas[indice_corte - horizonte])
    treino = df[df["data"] < limite_treino].copy()
    teste = df[df["data"] >= inicio_teste].copy()
    return treino, teste, inicio_teste
