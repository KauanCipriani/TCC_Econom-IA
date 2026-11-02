"""
Persistência baseada em arquivos.

O projeto não utiliza sistema gerenciador de banco de dados: as bases ficam
em arquivos Parquet (formato colunar comprimido, de leitura rápida no pandas)
e os modelos e métricas em arquivos binários e JSON na pasta "modelos".
"""

from __future__ import annotations

import pandas as pd

from econom_ia.config import CAMINHOS, Caminhos


class RepositorioDados:
    """Lê e grava as bases do projeto em formato Parquet."""

    def __init__(self, caminhos: Caminhos = CAMINHOS) -> None:
        self.caminhos = caminhos

    # ── OHLCV (dados brutos já limpos) ─────────────────────────────────────
    def carregar_ohlcv(self) -> pd.DataFrame:
        """Carrega a base de cotações. Lança FileNotFoundError se não existir."""
        return self._ler(self.caminhos.ohlcv)

    def salvar_ohlcv(self, df: pd.DataFrame) -> None:
        self._gravar(df, self.caminhos.ohlcv)

    # ── Base com indicadores e variável-alvo ────────────────────────────────
    def carregar_features(self) -> pd.DataFrame:
        return self._ler(self.caminhos.features)

    def salvar_features(self, df: pd.DataFrame) -> None:
        self._gravar(df, self.caminhos.features)

    # ── Previsões dos modelos nos conjuntos de avaliação ────────────────────
    def carregar_previsoes(self) -> pd.DataFrame:
        return self._ler(self.caminhos.previsoes_avaliacao)

    def salvar_previsoes(self, df: pd.DataFrame) -> None:
        self._gravar(df, self.caminhos.previsoes_avaliacao)

    # ── Utilitários ─────────────────────────────────────────────────────────
    @staticmethod
    def _ler(caminho) -> pd.DataFrame:
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        df = pd.read_parquet(caminho)
        df["data"] = pd.to_datetime(df["data"])
        df["ticker"] = df["ticker"].astype(str)
        return df.sort_values(["ticker", "data"]).reset_index(drop=True)

    @staticmethod
    def _gravar(df: pd.DataFrame, caminho) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        saida = df.copy()
        # Texto como "object" mantém o arquivo legível tanto no pandas 2
        # quanto no pandas 3 (que usa um tipo próprio para strings).
        for coluna in saida.columns:
            if saida[coluna].dtype == object or pd.api.types.is_string_dtype(saida[coluna]):
                saida[coluna] = saida[coluna].astype(object)
        saida.to_parquet(caminho, index=False)
