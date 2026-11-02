"""
Previsão de preço pela rede LSTM, com faixa de confiança.

A faixa usa os quantis de 10% e 90% dos erros observados pela LSTM no
conjunto de teste, para cada horizonte: é uma faixa de 80% estimada
empiricamente, sem supor que os erros sigam distribuição normal. Os erros
são expressos em múltiplos da volatilidade de 20 pregões, e a faixa é
reescalada pela volatilidade atual do ativo: fica mais larga em períodos
turbulentos e mais estreita em períodos calmos. Os limites inferior e
superior correspondem aos cenários pessimista e otimista do Painel Avançado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from econom_ia.config import CAMINHOS, Caminhos
from econom_ia.modelos import carregar_modelo
from econom_ia.servicos.calendario import proximos_pregoes


class ServicoPrevisao:
    def __init__(self, caminhos: Caminhos = CAMINHOS, metricas: dict | None = None) -> None:
        self.previsor = None
        self.erro: str | None = None
        self.faixas: dict = ((metricas or {}).get("regressao_lstm") or {}).get("faixas", {})
        try:
            self.previsor = carregar_modelo(caminhos.modelos / "lstm.pkl").previsor
        except FileNotFoundError:
            self.erro = "Modelo LSTM ainda não treinado (execute python treinar_modelos.py)."
        except ImportError:
            self.erro = "Biblioteca PyTorch não instalada (pip install torch)."
        except Exception as erro:
            self.erro = f"Não foi possível carregar a LSTM ({erro.__class__.__name__})."

    @property
    def disponivel(self) -> bool:
        return self.previsor is not None

    @property
    def horizonte_maximo(self) -> int:
        return self.previsor.horizontes if self.previsor else 15

    def prever(self, dados_ativo: pd.DataFrame, dias: int) -> pd.DataFrame:
        """
        Previsão para os próximos "dias" pregões de um ativo.

        Returns:
            DataFrame com data, previsao, lim_inf, lim_sup e variacao_pct
            (variação da previsão em relação ao último fechamento).
        """
        colunas = ["data", "previsao", "lim_inf", "lim_sup", "variacao_pct"]
        if not self.disponivel or dados_ativo.empty:
            return pd.DataFrame(columns=colunas)
        dias = int(min(max(dias, 1), self.horizonte_maximo))
        precos = self.previsor.prever_precos(dados_ativo, dias)
        if len(precos) == 0:
            return pd.DataFrame(columns=colunas)
        ultimo = float(dados_ativo["close"].iloc[-1])
        horizontes = np.arange(1, dias + 1)
        inferior, superior = self._limites_em_retorno(dados_ativo, horizontes)
        return pd.DataFrame({
            "data": proximos_pregoes(dados_ativo["data"].iloc[-1], dias),
            "previsao": precos,
            "lim_inf": precos * np.exp(inferior),
            "lim_sup": precos * np.exp(superior),
            "variacao_pct": (precos / ultimo - 1) * 100,
        })

    def _limites_em_retorno(self, dados_ativo: pd.DataFrame, horizontes: np.ndarray):
        """Desvios (em retorno) dos limites da faixa de 80% em cada horizonte."""
        faixas = [self.faixas.get(str(h), {}) for h in horizontes]
        volatilidade = dados_ativo["volatilidade_20"].dropna()
        if volatilidade.empty or not all("z10" in f for f in faixas):
            return (np.array([f.get("q10", 0.0) for f in faixas]),
                    np.array([f.get("q90", 0.0) for f in faixas]))
        escala = float(volatilidade.iloc[-1]) * np.sqrt(horizontes)
        return (np.array([f["z10"] for f in faixas]) * escala,
                np.array([f["z90"] for f in faixas]) * escala)
