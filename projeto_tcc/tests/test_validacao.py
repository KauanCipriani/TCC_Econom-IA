"""Divisões temporais: ordem cronológica e expurgo entre treino e validação."""

import numpy as np
import pytest

from econom_ia.avaliacao import DivisorTemporalPurgado, dividir_por_data


def test_validacao_cruzada_sem_sobreposicao(base_rotulada):
    datas = base_rotulada["data"].to_numpy()
    unicas = np.unique(datas)
    divisor = DivisorTemporalPurgado(n_folds=4, horizonte=5)
    dobras = list(divisor.split(base_rotulada, groups=datas))
    assert len(dobras) == divisor.get_n_splits() == 4
    for treino, validacao in dobras:
        ultima_treino = datas[treino].max()
        primeira_validacao = datas[validacao].min()
        # Entre o último dia de treino e o primeiro de validação há o expurgo de 5 pregões.
        distancia = np.searchsorted(unicas, primeira_validacao) - np.searchsorted(unicas, ultima_treino)
        assert distancia > 5


def test_validacao_exige_datas(base_rotulada):
    with pytest.raises(ValueError):
        next(DivisorTemporalPurgado().split(base_rotulada))


def test_divisao_treino_teste_com_expurgo(base_rotulada):
    treino, teste, inicio_teste = dividir_por_data(base_rotulada, 0.7, horizonte=5)
    assert treino["data"].max() < inicio_teste
    assert teste["data"].min() == inicio_teste
    unicas = np.sort(base_rotulada["data"].unique())
    expurgadas = unicas[(unicas >= treino["data"].max()) & (unicas < inicio_teste)]
    assert len(expurgadas) - 1 == 5
