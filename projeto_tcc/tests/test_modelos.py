"""Modelos: busca de hiperparâmetros, previsão e persistência (salvar e carregar)."""

import numpy as np
import pytest

from econom_ia.avaliacao import DivisorTemporalPurgado
from econom_ia.config import CLASSES
from econom_ia.features import CalculadoraIndicadores
from econom_ia.modelos import ModeloMLP, ModeloRandomForest, ModeloSVM, carregar_modelo

FEATURES = CalculadoraIndicadores.FEATURES_COMPLETAS


@pytest.mark.parametrize("classe", [ModeloRandomForest, ModeloSVM, ModeloMLP])
def test_classificadores(classe, base_rotulada, tmp_path):
    if classe is ModeloMLP:
        pytest.importorskip("torch")
    modelo = classe(FEATURES, semente=7, n_jobs=1)
    modelo.buscar_hiperparametros(base_rotulada, DivisorTemporalPurgado(3, 5), reduzida=True)
    assert modelo.resultado_busca["combinacoes_testadas"] == 1
    modelo.ajustar(base_rotulada)

    amostra = base_rotulada.tail(50)
    previsto = modelo.prever(amostra)
    probabilidades = modelo.prever_proba(amostra)
    assert set(previsto) <= set(CLASSES)
    assert list(probabilidades.columns) == list(CLASSES)
    assert np.allclose(probabilidades.sum(axis=1), 1.0, atol=1e-6)

    recarregado = carregar_modelo(modelo.salvar(tmp_path))
    assert list(recarregado.prever(amostra)) == list(previsto)


def test_lstm_previsao_e_classificacao(base_rotulada, tmp_path):
    pytest.importorskip("torch")
    from econom_ia.modelos import ModeloLSTMClassificacao, PrevisorLSTM

    previsor = PrevisorLSTM(janela=10, horizontes=5, oculto=8, epocas=2, tamanho_lote=64, semente=3)
    modelo = ModeloLSTMClassificacao(previsor, horizonte=5).ajustar(base_rotulada)
    assert modelo.corte_venda <= modelo.corte_compra

    classes = modelo.prever(base_rotulada)
    validas = [c for c in classes if c is not None]
    assert len(validas) > 0 and set(validas) <= set(CLASSES)

    ativo = base_rotulada[base_rotulada["ticker"] == "AAAA3"]
    precos = previsor.prever_precos(ativo, 5)
    assert len(precos) == 5 and np.all(precos > 0)

    recarregado = carregar_modelo(modelo.salvar(tmp_path))
    assert np.allclose(recarregado.previsor.prever_precos(ativo, 5), precos)
