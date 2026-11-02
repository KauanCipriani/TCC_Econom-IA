"""Métricas de classificação e de regressão e simulação de carteira."""

import numpy as np
import pandas as pd
import pytest

from econom_ia.avaliacao import AvaliadorClassificacao, AvaliadorRegressao, Backtester


def test_classificacao_perfeita():
    reais = ["venda", "neutro", "compra"] * 20
    resultado = AvaliadorClassificacao().avaliar(reais, reais)
    assert resultado["f1_macro"] == pytest.approx(1.0)
    assert resultado["mcc"] == pytest.approx(1.0)
    assert resultado["matriz_confusao"] == [[20, 0, 0], [0, 20, 0], [0, 0, 20]]


def test_intervalo_de_confianca_contem_a_estimativa():
    gerador = np.random.default_rng(1)
    classes = np.array(["venda", "neutro", "compra"], dtype=object)
    reais = gerador.choice(classes, 600)
    previstos = np.where(gerador.random(600) < 0.6, reais, gerador.choice(classes, 600))
    datas = np.repeat(pd.bdate_range("2024-01-01", periods=300), 2)
    avaliador = AvaliadorClassificacao()
    pontual = avaliador.avaliar(reais, previstos)
    intervalo = avaliador.intervalo_confianca(reais, previstos, datas, n_reamostragens=200)
    for chave, metrica in (("f1_macro", "f1_macro"), ("mcc", "mcc")):
        inferior, superior = intervalo[chave]
        assert inferior <= pontual[metrica] <= superior


def test_regressao_e_acerto_de_direcao():
    # Preço base 10: o real sobe, cai e sobe; o previsto sobe nos três casos.
    resultado = AvaliadorRegressao.avaliar([11, 9, 10.5], [10.5, 10.2, 10.5], [10, 10, 10])
    assert resultado["rmse"] == pytest.approx(np.sqrt((0.25 + 1.44 + 0) / 3))
    assert resultado["mae"] == pytest.approx(1.7 / 3)
    assert resultado["acerto_direcao"] == pytest.approx(2 / 3)


def test_backtest_sempre_comprado_equivale_a_comprar_e_manter():
    datas = pd.bdate_range("2024-01-01", periods=60)
    precos = 10 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, 60)))
    base = pd.DataFrame({"data": datas, "ticker": "X", "close": precos, "sinal": "compra"})
    resultado = Backtester(custo_operacao=0.0).simular(base)
    assert resultado.estrategia["retorno_total"] == pytest.approx(resultado.referencia["retorno_total"])
    assert resultado.estrategia["exposicao_media"] == pytest.approx(1.0)

    fora = Backtester().simular(base.assign(sinal="neutro"))
    assert fora.estrategia["retorno_total"] == pytest.approx(0.0)
    assert fora.estrategia["n_operacoes"] == 0
