"""
Fornecimento das cotações para a interface.

Tenta baixar o histórico atualizado no Yahoo Finance; sem conexão, usa o
arquivo Parquet local. Em ambos os casos aplica a mesma limpeza e o mesmo
cálculo de indicadores do treinamento, garantindo que os modelos recebam
variáveis calculadas exatamente da mesma forma.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from econom_ia.dados import ColetorYahoo, ErroColeta, LimpadorDados, RepositorioDados
from econom_ia.features import CalculadoraIndicadores


@dataclass
class CotacoesCarregadas:
    dados: pd.DataFrame      # OHLCV + indicadores de todos os ativos
    origem: str              # "online" (Yahoo Finance) ou "local" (arquivo Parquet)
    carregado_em: datetime

    @property
    def ultima_data(self) -> pd.Timestamp:
        return self.dados["data"].max()

    def ativo(self, ticker: str) -> pd.DataFrame:
        return self.dados[self.dados["ticker"] == ticker].reset_index(drop=True)


class ProvedorCotacoes:
    def __init__(self, repositorio: RepositorioDados | None = None,
                 coletor: ColetorYahoo | None = None) -> None:
        self.repositorio = repositorio or RepositorioDados()
        self.coletor = coletor or ColetorYahoo(inicio="2019-01-01")
        self.limpador = LimpadorDados()
        self.indicadores = CalculadoraIndicadores()

    def carregar(self, usar_internet: bool = True) -> CotacoesCarregadas:
        """
        Raises:
            FileNotFoundError: sem internet e sem arquivo local.
        """
        bruto, origem = None, "local"
        if usar_internet:
            try:
                bruto, origem = self.coletor.baixar(), "online"
            except ErroColeta:
                bruto = None
        if bruto is None:
            bruto = self.repositorio.carregar_ohlcv()
        limpo, _ = self.limpador.limpar(bruto)
        return CotacoesCarregadas(self.indicadores.calcular(limpo), origem, datetime.now())
