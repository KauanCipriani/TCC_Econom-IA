"""
Cálculo dos indicadores técnicos usados como variáveis de entrada.

Os indicadores são transformados em razões, retornos e z-scores. Dessa forma
ficam invariantes à escala de preço de cada ação (um RSI de 70 ou um preço 3%
acima da média significam a mesma coisa em qualquer ativo), o que permite
treinar um único modelo com várias ações e aplicá-lo a uma ação nunca vista.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CalculadoraIndicadores:
    """Calcula, para cada ativo, os indicadores técnicos e as variáveis do modelo."""

    # Variáveis utilizadas nos resultados parciais (TC I).
    FEATURES_ORIGINAIS: list[str] = [
        "rsi_14", "macd_hist_z",
        "sma_9_ratio", "sma_21_ratio", "sma_50_ratio",
        "ema_9_ratio", "ema_21_ratio",
        "bb_position",
    ]

    # Novas variáveis incluídas no TC II: volume normalizado, volatilidade
    # e retornos passados (defasagens do retorno). Assim como as originais,
    # são relativas: a volatilidade é comparada à de um período mais longo e
    # os retornos são divididos pela volatilidade recente, para que tenham o
    # mesmo significado em ativos e em regimes de mercado diferentes.
    FEATURES_NOVAS: list[str] = [
        "volume_z20",
        "vol_rel_10", "vol_rel_20", "atr_relativo",
        "ret_1_z", "ret_2_z", "ret_3_z", "ret_5_z", "ret_10_z",
    ]

    FEATURES_COMPLETAS: list[str] = FEATURES_ORIGINAIS + FEATURES_NOVAS

    # Nomes legíveis exibidos na interface ("qual indicador mais pesou").
    DESCRICOES: dict[str, str] = {
        "rsi_14": "RSI (14)",
        "macd_hist_z": "Histograma do MACD",
        "sma_9_ratio": "Preço x média móvel de 9",
        "sma_21_ratio": "Preço x média móvel de 21",
        "sma_50_ratio": "Preço x média móvel de 50",
        "ema_9_ratio": "Preço x média exponencial de 9",
        "ema_21_ratio": "Preço x média exponencial de 21",
        "bb_position": "Posição nas Bandas de Bollinger",
        "volume_z20": "Volume normalizado",
        "vol_rel_10": "Volatilidade de 10 pregões (relativa)",
        "vol_rel_20": "Volatilidade de 20 pregões (relativa)",
        "atr_relativo": "ATR (14) relativo",
        "ret_1_z": "Retorno do último pregão",
        "ret_2_z": "Retorno de 2 pregões",
        "ret_3_z": "Retorno de 3 pregões",
        "ret_5_z": "Retorno de 5 pregões",
        "ret_10_z": "Retorno de 10 pregões",
    }

    def calcular(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recebe OHLCV de um ou mais ativos e devolve a base com indicadores."""
        partes = [
            self._calcular_ativo(grupo)
            for _, grupo in df.sort_values(["ticker", "data"]).groupby("ticker", sort=False)
        ]
        return pd.concat(partes, ignore_index=True)

    def _calcular_ativo(self, grupo: pd.DataFrame) -> pd.DataFrame:
        g = grupo.reset_index(drop=True).copy()
        fechamento = g["close"]

        # RSI de 14 períodos (médias simples de ganhos e perdas). Sem perdas na
        # janela o RSI vale 100; sem ganhos nem perdas (preço parado), 50.
        variacao = fechamento.diff()
        ganho = variacao.clip(lower=0).rolling(14).mean()
        perda = (-variacao.clip(upper=0)).rolling(14).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            rsi = 100 - 100 / (1 + ganho / perda)
        g["rsi_14"] = rsi.mask((ganho == 0) & (perda == 0), 50.0)

        # MACD (12, 26, 9) e z-score do histograma em janela de 60 pregões.
        ema_12 = fechamento.ewm(span=12, adjust=False).mean()
        ema_26 = fechamento.ewm(span=26, adjust=False).mean()
        g["macd_line"] = ema_12 - ema_26
        g["macd_signal"] = g["macd_line"].ewm(span=9, adjust=False).mean()
        g["macd_hist"] = g["macd_line"] - g["macd_signal"]
        media_hist = g["macd_hist"].rolling(60).mean()
        desvio_hist = g["macd_hist"].rolling(60).std()
        g["macd_hist_z"] = (g["macd_hist"] - media_hist) / desvio_hist.replace(0, np.nan)

        # Médias móveis simples e exponenciais, e a distância do preço a elas.
        for periodo in (9, 21, 50):
            g[f"sma_{periodo}"] = fechamento.rolling(periodo).mean()
            g[f"ema_{periodo}"] = fechamento.ewm(span=periodo, adjust=False).mean()
            g[f"sma_{periodo}_ratio"] = fechamento / g[f"sma_{periodo}"] - 1
        for periodo in (9, 21):
            g[f"ema_{periodo}_ratio"] = fechamento / g[f"ema_{periodo}"] - 1

        # Bandas de Bollinger (20 períodos, 2 desvios) e a posição do preço
        # entre elas (0 = banda inferior, 1 = banda superior).
        media_20 = fechamento.rolling(20).mean()
        desvio_20 = fechamento.rolling(20).std()
        g["bb_middle"] = media_20
        g["bb_upper"] = media_20 + 2 * desvio_20
        g["bb_lower"] = media_20 - 2 * desvio_20
        amplitude = (g["bb_upper"] - g["bb_lower"]).replace(0, np.nan)
        g["bb_position"] = (fechamento - g["bb_lower"]) / amplitude

        # Retornos logarítmicos passados (defasagens do retorno).
        log_preco = np.log(fechamento)
        for k in (1, 2, 3, 5, 10):
            g[f"ret_{k}"] = log_preco - log_preco.shift(k)

        # Volatilidade: desvio-padrão dos retornos diários. As versões
        # relativas comparam a volatilidade recente com a de um período longo
        # (acima de 1 = mercado mais agitado que o habitual para o ativo).
        for janela in (10, 20, 60, 120):
            g[f"volatilidade_{janela}"] = g["ret_1"].rolling(janela).std()
        g["vol_rel_10"] = g["volatilidade_10"] / g["volatilidade_60"]
        g["vol_rel_20"] = g["volatilidade_20"] / g["volatilidade_120"]

        # Retornos padronizados pela volatilidade de 20 pregões (um retorno de
        # 2% é grande em um período calmo e pequeno em um período turbulento).
        for k in (1, 2, 3, 5, 10):
            g[f"ret_{k}_z"] = g[f"ret_{k}"] / (g["volatilidade_20"] * np.sqrt(k))

        # ATR de 14 períodos (média da amplitude verdadeira) relativo ao preço
        # e comparado à sua média de 120 pregões.
        fechamento_anterior = fechamento.shift(1)
        amplitude_verdadeira = pd.concat(
            [
                g["high"] - g["low"],
                (g["high"] - fechamento_anterior).abs(),
                (g["low"] - fechamento_anterior).abs(),
            ],
            axis=1,
        ).max(axis=1)
        g["atr_14_ratio"] = amplitude_verdadeira.rolling(14).mean() / fechamento
        g["atr_relativo"] = g["atr_14_ratio"] / g["atr_14_ratio"].rolling(120).mean()

        # Volume normalizado: z-score do log do volume em janela de 20 pregões.
        log_volume = np.log(g["volume"].clip(lower=1).astype(float))
        g["volume_z20"] = (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std()

        return g
