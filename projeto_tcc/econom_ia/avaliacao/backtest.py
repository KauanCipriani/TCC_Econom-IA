"""
Simulação simplificada de uma estratégia guiada pelos sinais do modelo.

Regra: ao fim de cada pregão, o investidor fica comprado no ativo se o sinal
for "compra" e fora do mercado caso contrário. As posições nas ações têm o
mesmo peso. A referência é comprar e manter as mesmas ações com pesos iguais.
Há um custo por mudança de posição; juros do caixa e impostos são ignorados.
A simulação serve para avaliar se os sinais agregam informação econômica, e
não como promessa de rentabilidade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PREGOES_POR_ANO = 252


@dataclass
class ResultadoBacktest:
    """Curvas de capital (base 1) e métricas da estratégia e da referência."""

    curva: pd.DataFrame
    estrategia: dict
    referencia: dict


class Backtester:
    """Executa a simulação a partir de uma base com data, ticker, close e sinal."""

    def __init__(self, custo_operacao: float = 0.0005) -> None:
        # 0,05% por entrada ou saída, uma aproximação de corretagem e emolumentos.
        self.custo_operacao = custo_operacao

    def simular(self, df: pd.DataFrame) -> ResultadoBacktest:
        partes = []
        for _, grupo in df.sort_values(["ticker", "data"]).groupby("ticker", sort=False):
            g = grupo.copy()
            g["retorno_dia_seguinte"] = g["close"].shift(-1) / g["close"] - 1
            g["posicao"] = (g["sinal"] == "compra").astype(float)
            g["operacoes"] = g["posicao"].diff().abs().fillna(g["posicao"])
            g["retorno_estrategia"] = (
                g["posicao"] * g["retorno_dia_seguinte"] - self.custo_operacao * g["operacoes"]
            )
            partes.append(g.dropna(subset=["retorno_dia_seguinte"]))
        base = pd.concat(partes)

        diario = base.groupby("data").agg(
            estrategia=("retorno_estrategia", "mean"),
            referencia=("retorno_dia_seguinte", "mean"),
            exposicao=("posicao", "mean"),
        )
        curva = (1 + diario[["estrategia", "referencia"]]).cumprod().reset_index()
        return ResultadoBacktest(
            curva=curva,
            estrategia=self._metricas(diario["estrategia"], diario["exposicao"].mean(), int(base["operacoes"].sum())),
            referencia=self._metricas(diario["referencia"], 1.0, len(df["ticker"].unique())),
        )

    @staticmethod
    def _metricas(retornos: pd.Series, exposicao: float, operacoes: int) -> dict:
        capital = (1 + retornos).cumprod()
        n = len(retornos)
        retorno_total = float(capital.iloc[-1] - 1)
        volatilidade = float(retornos.std() * np.sqrt(PREGOES_POR_ANO))
        media_anual = float(retornos.mean() * PREGOES_POR_ANO)
        return {
            "retorno_total": retorno_total,
            "retorno_anualizado": float((1 + retorno_total) ** (PREGOES_POR_ANO / n) - 1),
            "volatilidade_anual": volatilidade,
            "indice_sharpe": media_anual / volatilidade if volatilidade > 0 else 0.0,
            "max_drawdown": float((capital / capital.cummax() - 1).min()),
            "exposicao_media": float(exposicao),
            "n_operacoes": int(operacoes),
            "n_pregoes": int(n),
        }
