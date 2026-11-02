"""Contexto compartilhado pelas telas em cada execução da interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
import streamlit as st

from econom_ia.interface import dados_app
from econom_ia.servicos import CentralAlertas, CotacoesCarregadas, ServicoPrevisao, ServicoSinais
from econom_ia.servicos.sinais import NOMES_MODELOS


@dataclass
class ContextoApp:
    """Dados e serviços que as telas utilizam."""

    cotacoes: CotacoesCarregadas
    servico_sinais: ServicoSinais
    servico_previsao: ServicoPrevisao
    usar_internet: bool

    # Páginas registradas no menu lateral (preenchido por AplicacaoEconomIA).
    paginas: ClassVar[dict] = {}

    # ── Informações do treinamento ─────────────────────────────────────────
    @property
    def metricas(self) -> dict | None:
        return self.servico_sinais.metricas

    @property
    def modelo_principal(self) -> str:
        return self.servico_sinais.principal

    @property
    def horizonte(self) -> int:
        return self.servico_sinais.horizonte

    @property
    def inicio_prospectivo(self) -> pd.Timestamp | None:
        """Primeiro pregão após o período usado no treinamento."""
        periodo = ((self.metricas or {}).get("info") or {}).get("periodo_prospectivo")
        return pd.Timestamp(periodo[0]) if periodo else None

    @staticmethod
    def nome_modelo(chave: str) -> str:
        return NOMES_MODELOS.get(chave, chave)

    def modelos_disponiveis(self) -> dict[str, str]:
        return self.servico_sinais.disponiveis()

    # ── Dados com sinais ───────────────────────────────────────────────────
    def dados_com_sinais(self, chave_modelo: str | None = None) -> pd.DataFrame:
        """Cotações de todos os ativos com o sinal do modelo escolhido."""
        chave = chave_modelo or self.modelo_principal
        versao = f"{self.cotacoes.origem}-{self.cotacoes.carregado_em:%Y%m%d%H%M%S}"
        return dados_app.sinais(chave, self.usar_internet, versao)

    def ativo(self, ticker: str, chave_modelo: str | None = None) -> pd.DataFrame:
        """Histórico de um ativo (com indicadores e sinais), em ordem cronológica."""
        dados = self.dados_com_sinais(chave_modelo)
        return dados[dados["ticker"] == ticker].sort_values("data").reset_index(drop=True)

    def alertas(self, chave_modelo: str | None = None, confianca_minima: float = 0.0,
                inicio: pd.Timestamp | None = None) -> pd.DataFrame:
        """Histórico de alertas de todos os ativos para o modelo escolhido."""
        central = CentralAlertas(self.horizonte, confianca_minima)
        return central.historico(self.dados_com_sinais(chave_modelo), inicio)

    # ── Navegação ──────────────────────────────────────────────────────────
    @classmethod
    def navegar(cls, pagina: str) -> None:
        """Abre outra tela, como se o usuário tivesse clicado nela no menu lateral."""
        if pagina in cls.paginas:
            st.switch_page(cls.paginas[pagina])
