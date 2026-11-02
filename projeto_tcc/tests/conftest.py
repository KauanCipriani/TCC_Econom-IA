"""
Configuração compartilhada dos testes automatizados (pytest).

Os testes unitários usam uma base pequena gerada por passeio aleatório, que
exercita os cálculos sem depender de internet nem dos arquivos do projeto. A
base real de cotações é usada apenas nos testes da interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from econom_ia.features import CalculadoraIndicadores, RegraLimiar, Rotulador  # noqa: E402


def gerar_ohlcv(tickers: tuple[str, ...] = ("AAAA3", "BBBB4"), dias: int = 420, semente: int = 0) -> pd.DataFrame:
    """Cotações artificiais com o mesmo formato da base real (uso exclusivo nos testes)."""
    gerador = np.random.default_rng(semente)
    datas = pd.bdate_range("2021-01-04", periods=dias)
    partes = []
    for i, ticker in enumerate(tickers):
        retornos = gerador.normal(0.0003, 0.018, dias)
        fechamento = 20.0 * (1 + i) * np.exp(np.cumsum(retornos))
        abertura = fechamento * (1 + gerador.normal(0, 0.004, dias))
        maxima = np.maximum(fechamento, abertura) * (1 + np.abs(gerador.normal(0, 0.006, dias)))
        minima = np.minimum(fechamento, abertura) * (1 - np.abs(gerador.normal(0, 0.006, dias)))
        partes.append(pd.DataFrame({
            "data": datas, "ticker": ticker, "open": abertura, "high": maxima, "low": minima,
            "close": fechamento, "volume": gerador.integers(1_000_000, 5_000_000, dias),
        }))
    return pd.concat(partes, ignore_index=True)


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    return gerar_ohlcv()


@pytest.fixture(scope="session")
def base_indicadores(ohlcv) -> pd.DataFrame:
    return CalculadoraIndicadores().calcular(ohlcv)


@pytest.fixture(scope="session")
def base_rotulada(base_indicadores) -> pd.DataFrame:
    rotulada = Rotulador(5, RegraLimiar("fixo", 0.02)).adicionar_alvo(base_indicadores)
    colunas = [*CalculadoraIndicadores.FEATURES_COMPLETAS, "alvo"]
    return rotulada.dropna(subset=colunas).reset_index(drop=True)
