"""
Coleta e limpeza das cotações diárias (OHLCV) das ações monitoradas.

A coleta usa a biblioteca yfinance, que acessa a API pública do Yahoo Finance.
Os preços são baixados já ajustados por proventos e desdobramentos
(auto_adjust=True), de modo que a série de retornos não apresente "quedas"
artificiais nas datas ex-dividendos ou de bonificação.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from econom_ia.config import ATIVOS, Ativo


class ErroColeta(RuntimeError):
    """Nenhum dado pôde ser obtido da fonte externa (sem conexão, por exemplo)."""


class ColetorYahoo:
    """Baixa o histórico diário de preços e volume no Yahoo Finance."""

    COLUNAS = ["data", "ticker", "open", "high", "low", "close", "volume"]

    def __init__(
        self,
        ativos: Iterable[Ativo] | None = None,
        inicio: str = "2019-01-01",
        fim: str | None = None,
    ) -> None:
        """
        Args:
            ativos: ações a coletar (padrão: as quatro do cadastro).
            inicio: primeira data da coleta (AAAA-MM-DD).
            fim: última data da coleta, inclusiva. None = até hoje.
        """
        self.ativos = list(ativos) if ativos is not None else list(ATIVOS.values())
        self.inicio = inicio
        self.fim = fim

    def baixar(self) -> pd.DataFrame:
        """
        Retorna um DataFrame com as colunas de COLUNAS para todos os ativos.

        Raises:
            ErroColeta: quando nenhum ativo pôde ser baixado.
        """
        try:
            import yfinance as yf  # importação tardia: só é necessária aqui
        except ImportError as erro:
            raise ErroColeta("Biblioteca yfinance não instalada.") from erro

        # No yfinance a data final é exclusiva; somamos um dia para incluí-la.
        fim_exclusivo = None
        if self.fim is not None:
            fim_exclusivo = (pd.Timestamp(self.fim) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        quadros = []
        for ativo in self.ativos:
            try:
                bruto = yf.download(
                    ativo.simbolo_yahoo,
                    start=self.inicio,
                    end=fim_exclusivo,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception:  # falha de rede em um ativo não interrompe os demais
                continue
            if bruto is None or bruto.empty:
                continue
            quadros.append(self._padronizar(bruto, ativo.ticker))

        if not quadros:
            raise ErroColeta("Não foi possível baixar cotações do Yahoo Finance.")
        df = pd.concat(quadros, ignore_index=True)
        return df.sort_values(["ticker", "data"]).reset_index(drop=True)

    @classmethod
    def _padronizar(cls, bruto: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Converte a saída do yfinance para o formato interno do projeto."""
        bruto = bruto.copy()
        # Com um único símbolo, versões recentes do yfinance devolvem colunas
        # em dois níveis (campo, símbolo); mantemos apenas o nome do campo.
        if isinstance(bruto.columns, pd.MultiIndex):
            bruto.columns = bruto.columns.get_level_values(0)
        bruto = bruto.reset_index().rename(
            columns={
                "Date": "data", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            }
        )
        datas = pd.to_datetime(bruto["data"])
        if getattr(datas.dt, "tz", None) is not None:
            datas = datas.dt.tz_localize(None)
        bruto["data"] = datas.dt.normalize()
        bruto["ticker"] = ticker
        return bruto[cls.COLUNAS]


@dataclass
class RelatorioLimpeza:
    """Resumo do que foi alterado pela limpeza (registrado no treinamento)."""

    registros_entrada: int = 0
    duplicados_removidos: int = 0
    ausentes_removidos: int = 0
    sem_negociacao_removidos: int = 0
    outliers_marcados: int = 0
    registros_saida: int = 0
    datas_sem_negociacao: list[str] = field(default_factory=list)

    def como_dict(self) -> dict:
        return dict(self.__dict__)


class LimpadorDados:
    """
    Trata valores ausentes e identifica outliers nas séries de preço.

    Regras:
      1. registros duplicados (mesmo ativo e data) são removidos;
      2. registros sem preço (ausentes) são removidos;
      3. registros com volume zero são removidos, pois correspondem a dias em
         que a fonte repetiu o último preço sem ter havido negociação;
      4. retornos diários extremos são apenas marcados (coluna outlier_flag),
         e não removidos, porque refletem eventos reais de mercado (por
         exemplo, o choque da Covid-19 em março de 2020).
    """

    def __init__(self, limite_z_robusto: float = 5.0) -> None:
        # Um retorno é marcado como outlier quando o seu z-score robusto
        # (baseado na mediana e no desvio absoluto mediano) supera o limite.
        self.limite_z_robusto = limite_z_robusto

    def limpar(self, df: pd.DataFrame) -> tuple[pd.DataFrame, RelatorioLimpeza]:
        relatorio = RelatorioLimpeza(registros_entrada=len(df))
        dados = df.copy()
        dados["data"] = pd.to_datetime(dados["data"])

        antes = len(dados)
        dados = dados.drop_duplicates(subset=["ticker", "data"], keep="last")
        relatorio.duplicados_removidos = antes - len(dados)

        antes = len(dados)
        dados = dados.dropna(subset=["open", "high", "low", "close"])
        relatorio.ausentes_removidos = antes - len(dados)

        sem_negocio = dados["volume"].fillna(0) <= 0
        relatorio.sem_negociacao_removidos = int(sem_negocio.sum())
        relatorio.datas_sem_negociacao = sorted(
            f"{t} {d:%Y-%m-%d}" for t, d in dados.loc[sem_negocio, ["ticker", "data"]].itertuples(index=False)
        )
        dados = dados.loc[~sem_negocio]

        dados = dados.sort_values(["ticker", "data"]).reset_index(drop=True)
        dados["outlier_flag"] = self._marcar_outliers(dados)
        relatorio.outliers_marcados = int(dados["outlier_flag"].sum())
        relatorio.registros_saida = len(dados)
        return dados, relatorio

    def _marcar_outliers(self, dados: pd.DataFrame) -> pd.Series:
        marcas = pd.Series(False, index=dados.index)
        for _, grupo in dados.groupby("ticker", sort=False):
            retorno = np.log(grupo["close"]).diff()
            mediana = retorno.median()
            mad = (retorno - mediana).abs().median()
            if not mad or np.isnan(mad):
                continue
            z_robusto = (retorno - mediana) / (1.4826 * mad)
            marcas.loc[grupo.index] = z_robusto.abs() > self.limite_z_robusto
        return marcas
