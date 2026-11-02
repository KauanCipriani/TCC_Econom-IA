"""Indicadores técnicos: faixas válidas, fórmulas e ausência de vazamento temporal."""

import numpy as np
import pandas as pd

from econom_ia.features import CalculadoraIndicadores


def test_todas_as_variaveis_sao_calculadas(base_indicadores):
    for coluna in CalculadoraIndicadores.FEATURES_COMPLETAS:
        assert coluna in base_indicadores.columns
        assert base_indicadores[coluna].notna().sum() > 200


def test_faixas_e_formulas(base_indicadores):
    dados = base_indicadores.dropna(subset=CalculadoraIndicadores.FEATURES_COMPLETAS)
    assert dados["rsi_14"].between(0, 100).all()
    assert np.allclose(dados["sma_21_ratio"], dados["close"] / dados["sma_21"] - 1)
    posicao = (dados["close"] - dados["bb_lower"]) / (dados["bb_upper"] - dados["bb_lower"])
    assert np.allclose(dados["bb_position"], posicao)
    assert (dados["vol_rel_20"] > 0).all()


def test_indicadores_usam_apenas_o_passado(ohlcv):
    """O valor de um dia não pode mudar quando se acrescentam dias futuros."""
    calculadora = CalculadoraIndicadores()
    completo = calculadora.calcular(ohlcv)
    corte = ohlcv["data"].sort_values().unique()[300]
    parcial = calculadora.calcular(ohlcv[ohlcv["data"] <= corte])
    colunas = CalculadoraIndicadores.FEATURES_COMPLETAS
    antes = completo[completo["data"] == corte].set_index("ticker")[colunas]
    depois = parcial[parcial["data"] == corte].set_index("ticker")[colunas]
    pd.testing.assert_frame_equal(antes.sort_index(), depois.sort_index())


def test_rsi_em_casos_extremos():
    datas = pd.bdate_range("2024-01-01", periods=40)
    precos = np.r_[np.linspace(10, 20, 20), np.full(20, 20.0)]  # só altas e depois preço parado
    base = pd.DataFrame({"data": datas, "ticker": "X", "open": precos, "high": precos, "low": precos,
                         "close": precos, "volume": 1000})
    rsi = CalculadoraIndicadores().calcular(base)["rsi_14"]
    assert rsi.iloc[19] == 100.0   # sem perdas na janela
    assert rsi.iloc[-1] == 50.0    # sem ganhos nem perdas
