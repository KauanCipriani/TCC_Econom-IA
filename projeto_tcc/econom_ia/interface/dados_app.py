"""
Carregamento dos dados da interface, com o cache do Streamlit.

  * cache_resource guarda objetos pesados que não mudam (modelos treinados);
  * cache_data guarda resultados de consultas e expira após "ttl" segundos,
    de modo que cotações (15 min), manchetes (30 min) e indicadores
    macroeconômicos (1 h) se renovam sozinhos sem reiniciar o aplicativo.

Falhas de rede nunca interrompem a interface: cada função devolve None e a
tela correspondente exibe a informação como indisponível.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from econom_ia.dados import RepositorioDados
from econom_ia.integracoes import (
    ClienteBCB,
    ClienteFocus,
    ClienteFRED,
    ClienteMercado,
    ClienteNoticias,
    ClientePolymarket,
    ErroIntegracao,
    Manchetes,
)
from econom_ia.servicos import ProvedorCotacoes, ServicoPrevisao, ServicoSinais


@st.cache_resource(show_spinner=False)
def servico_sinais() -> ServicoSinais:
    return ServicoSinais()


@st.cache_resource(show_spinner=False)
def servico_previsao() -> ServicoPrevisao:
    return ServicoPrevisao(metricas=servico_sinais().metricas)


@st.cache_data(ttl=900, show_spinner="Atualizando as cotações da B3...")
def cotacoes(usar_internet: bool = True):
    return ProvedorCotacoes().carregar(usar_internet)


@st.cache_data(ttl=900, show_spinner="Gerando os sinais do modelo...")
def sinais(chave_modelo: str, usar_internet: bool, versao_dados: str) -> pd.DataFrame:
    """Cotações com o sinal do modelo; "versao_dados" invalida o cache quando os dados mudam."""
    return servico_sinais().gerar(cotacoes(usar_internet).dados, chave_modelo)


@st.cache_data(ttl=3600, show_spinner=False)
def serie_bcb(chave: str, usar_internet: bool = True, anos: int = 5):
    try:
        return ClienteBCB(validade_s=3600, offline=not usar_internet).serie(chave, anos)
    except (ErroIntegracao, KeyError, ValueError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def serie_fred(chave: str, usar_internet: bool = True, anos: int = 5):
    try:
        return ClienteFRED(validade_s=3600, offline=not usar_internet).serie(chave, anos)
    except (ErroIntegracao, KeyError, ValueError):
        return None


@st.cache_data(ttl=900, show_spinner=False)
def cotacao_mercado(chave: str, usar_internet: bool = True):
    return ClienteMercado(validade_s=900, offline=not usar_internet).cotacao(chave)


@st.cache_data(ttl=3600, show_spinner=False)
def expectativas_focus(usar_internet: bool = True):
    """Projeções do Relatório Focus para a Selic nas próximas reuniões do Copom."""
    try:
        return ClienteFocus(validade_s=3600, offline=not usar_internet).expectativas_selic()
    except (ErroIntegracao, KeyError, ValueError):
        return None


@st.cache_data(ttl=900, show_spinner=False)
def decisao_copom(usar_internet: bool = True):
    """
    Evento do Polymarket para a próxima reunião do Copom.

    Returns:
        (mercado ou None, consulta_realizada). consulta_realizada é False quando
        a API não respondeu, o que permite à tela diferenciar "sem mercado
        ativo" de "fonte inacessível".
    """
    try:
        return ClientePolymarket(validade_s=900, offline=not usar_internet).proxima_decisao_copom(), True
    except ErroIntegracao:
        return None, False
    except (KeyError, ValueError):
        return None, True


@st.cache_data(ttl=1800, show_spinner=False)
def manchetes(usar_internet: bool = True) -> Manchetes:
    try:
        return ClienteNoticias(validade_s=1800, offline=not usar_internet).manchetes()
    except (ErroIntegracao, ValueError):
        return Manchetes([], None, None)


@st.cache_data(show_spinner=False)
def previsoes_avaliacao(data_treinamento: str) -> pd.DataFrame | None:
    """Previsões salvas pelo treinamento ("data_treinamento" renova o cache)."""
    try:
        return RepositorioDados().carregar_previsoes()
    except (OSError, ValueError, KeyError):
        return None


def limpar_cache() -> None:
    """
    Força a atualização de todos os dados na próxima execução.

    Também recarrega os modelos, o que permite usar um treinamento novo
    (python treinar_modelos.py) sem reiniciar a interface.
    """
    st.cache_data.clear()
    st.cache_resource.clear()
