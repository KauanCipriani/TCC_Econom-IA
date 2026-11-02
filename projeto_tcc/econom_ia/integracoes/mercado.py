"""
Cotações de referência do mercado (Ibovespa e dólar) via Yahoo Finance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from econom_ia.integracoes.http import ClienteHTTP, ErroIntegracao


@dataclass
class Cotacao:
    """Último fechamento e variação em relação ao pregão anterior."""

    nome: str
    valor: float
    variacao_pct: float
    data: str
    origem: str
    obtido_em: datetime


class ClienteMercado(ClienteHTTP):
    nome = "mercado"
    SIMBOLOS = {"ibovespa": "^BVSP", "dolar": "BRL=X"}

    def cotacao(self, chave: str) -> Cotacao | None:
        """Cotação de "ibovespa" ou "dolar"; None se indisponível."""
        simbolo = self.SIMBOLOS[chave]
        try:
            resposta = self.consultar(f"{self.nome}_{chave}", lambda: self._baixar(simbolo))
        except ErroIntegracao:
            return None
        dados = resposta.conteudo
        return Cotacao(chave, dados["valor"], dados["variacao_pct"], dados["data"],
                       resposta.origem, resposta.obtido_em)

    @staticmethod
    def _baixar(simbolo: str) -> dict:
        import yfinance as yf

        historico = yf.download(simbolo, period="10d", interval="1d", auto_adjust=True,
                                progress=False, threads=False)
        if historico is None or historico.empty:
            raise ValueError("sem dados")
        if isinstance(historico.columns, pd.MultiIndex):
            historico.columns = historico.columns.get_level_values(0)
        fechamentos = historico["Close"].dropna()
        if len(fechamentos) < 2:
            raise ValueError("histórico insuficiente")
        atual, anterior = float(fechamentos.iloc[-1]), float(fechamentos.iloc[-2])
        return {
            "valor": atual,
            "variacao_pct": (atual / anterior - 1) * 100,
            "data": pd.Timestamp(fechamentos.index[-1]).strftime("%Y-%m-%d"),
        }
