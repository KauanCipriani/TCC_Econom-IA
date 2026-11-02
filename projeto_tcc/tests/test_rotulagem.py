"""Variável-alvo: regra de limiar, alinhamento temporal e rótulos ausentes."""

import numpy as np
import pytest

from econom_ia.features import RegraLimiar, Rotulador


def test_classificacao_com_limiar_fixo():
    rotulador = Rotulador(5, RegraLimiar("fixo", 0.02))
    classes = rotulador.classificar([0.03, -0.03, 0.01, 0.02, np.nan], 0.02)
    assert list(classes) == ["compra", "venda", "neutro", "neutro", None]


def test_retorno_futuro_por_ativo(base_indicadores):
    rotulada = Rotulador(5, RegraLimiar("fixo", 0.015)).adicionar_alvo(base_indicadores)
    for _, grupo in rotulada.groupby("ticker"):
        esperado = grupo["close"].shift(-5) / grupo["close"] - 1
        assert np.allclose(grupo["retorno_futuro"], esperado, equal_nan=True)
        # Os últimos 5 pregões de cada ativo ainda não têm futuro conhecido.
        assert grupo["alvo"].tail(5).isna().all()
        assert grupo["alvo"].iloc[:-5].notna().all()


def test_limiar_adaptativo_depende_da_volatilidade(base_indicadores):
    regra = RegraLimiar("volatilidade", 0.5)
    limiares = Rotulador(5, regra).limiares(base_indicadores)
    esperado = 0.5 * base_indicadores["volatilidade_20"].to_numpy() * np.sqrt(5)
    assert np.allclose(limiares, esperado, equal_nan=True)
    with pytest.raises(ValueError):
        Rotulador(5, regra).limiares(base_indicadores.drop(columns="volatilidade_20"))


def test_regra_valida_e_descricao():
    assert RegraLimiar("fixo", 0.025).descricao() == "±2,5%"
    assert RegraLimiar.de_dict(RegraLimiar("volatilidade", 0.5).como_dict()) == RegraLimiar("volatilidade", 0.5)
    with pytest.raises(ValueError):
        RegraLimiar("outro", 0.01)
    with pytest.raises(ValueError):
        RegraLimiar("fixo", 0)


def test_distribuicao_soma_um(base_rotulada):
    distribuicao = Rotulador.distribuicao(base_rotulada["alvo"])
    assert set(distribuicao) == {"venda", "neutro", "compra"}
    assert sum(distribuicao.values()) == pytest.approx(1.0, abs=1e-3)
