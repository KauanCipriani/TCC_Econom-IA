"""Validação temporal, métricas de desempenho e simulação de estratégia."""

from econom_ia.avaliacao.backtest import Backtester
from econom_ia.avaliacao.metricas import AvaliadorClassificacao, AvaliadorRegressao
from econom_ia.avaliacao.validacao import DivisorTemporalPurgado, dividir_por_data

__all__ = [
    "AvaliadorClassificacao",
    "AvaliadorRegressao",
    "Backtester",
    "DivisorTemporalPurgado",
    "dividir_por_data",
]
