"""
Classe base das telas da interface.

Cada tela é uma subclasse de Tela que informa como aparece no menu lateral
(chave, rótulo e ícone) e implementa renderizar(). Os seletores e
utilidades usados por mais de uma tela ficam aqui, evitando repetição.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from econom_ia.config import ATIVOS
from econom_ia.interface import formatacao as fmt
from econom_ia.interface.componentes import Componentes
from econom_ia.interface.contexto import ContextoApp
from econom_ia.interface.graficos import FabricaGraficos

# Janelas de exibição dos gráficos de preço (em pregões; None = todo o histórico).
PERIODOS_GRAFICO: dict[str, int | None] = {
    "1 mês": 22,
    "6 meses": 126,
    "1 ano": 252,
    "Desde 2019": None,
}


class Tela(ABC):
    """Uma página do Econom-IA."""

    chave: ClassVar[str]      # identificador usado na navegação
    rotulo: ClassVar[str]     # texto da opção no menu lateral
    # Blocos que o usuário pode esconder em "Personalizar tela" (chave -> nome exibido).
    # A resposta principal de cada tela e o aviso legal ficam sempre visíveis.
    SECOES_OPCIONAIS: ClassVar[dict[str, str]] = {}
    icone: ClassVar[str]      # ícone do botão (Material Symbols)

    def __init__(self, contexto: ContextoApp) -> None:
        self.ctx = contexto
        self.ui = Componentes
        self.graficos = FabricaGraficos

    @abstractmethod
    def renderizar(self) -> None:
        """Desenha a tela."""

    # ── Cabeçalho e personalização ─────────────────────────────────────────
    def cabecalho(self, titulo: str, subtitulo: str = "") -> None:
        """Título da tela à esquerda e o botão "Personalizar tela" à direita."""
        if not self.SECOES_OPCIONAIS:
            self.ui.renderizar(self.ui.cabecalho(titulo, subtitulo))
            return
        texto, botao = st.columns([5, 1.25], vertical_alignment="center")
        with texto:
            self.ui.renderizar(self.ui.cabecalho(titulo, subtitulo))
        with botao:
            self.personalizar()

    def personalizar(self) -> None:
        """
        Chaves para mostrar ou esconder os blocos opcionais da tela.

        As escolhas ficam guardadas na sessão: continuam valendo ao trocar de
        tela e voltar, e o botão "Mostrar tudo" desfaz a personalização.
        """
        if not self.SECOES_OPCIONAIS:
            return
        preferencias = st.session_state.setdefault("preferencias_telas", {}).setdefault(self.chave, {})
        escondidos = sum(1 for secao in self.SECOES_OPCIONAIS if not preferencias.get(secao, True))
        rotulo = "Personalizar tela" + (f" ({escondidos} oculto{'s' if escondidos > 1 else ''})" if escondidos else "")
        with st.popover(rotulo, icon=":material/tune:", width="stretch"):
            st.markdown("**O que mostrar nesta tela**")
            st.caption("Esconda o que não quiser ver agora. A resposta principal fica sempre visível.")
            for secao, nome in self.SECOES_OPCIONAIS.items():
                chave_widget = f"mostrar_{self.chave}_{secao}"

                def lembrar(secao: str = secao, chave_widget: str = chave_widget) -> None:
                    preferencias[secao] = st.session_state[chave_widget]

                if chave_widget not in st.session_state:
                    st.session_state[chave_widget] = preferencias.get(secao, True)
                st.toggle(nome, key=chave_widget, on_change=lembrar)

            def mostrar_tudo() -> None:
                for secao in self.SECOES_OPCIONAIS:
                    preferencias[secao] = True
                    st.session_state[f"mostrar_{self.chave}_{secao}"] = True

            st.button("Mostrar tudo", key=f"mostrar_tudo_{self.chave}", on_click=mostrar_tudo,
                      disabled=not escondidos, width="stretch")

    def mostrar(self, secao: str) -> bool:
        """Se o bloco deve aparecer (todos aparecem até o usuário escondê-los)."""
        preferencias = st.session_state.get("preferencias_telas", {}).get(self.chave, {})
        return preferencias.get(secao, True)

    # ── Utilidades comuns ──────────────────────────────────────────────────
    @staticmethod
    def exibir(figura: go.Figure, chave: str | None = None) -> None:
        """Exibe um gráfico Plotly ocupando a largura disponível."""
        st.plotly_chart(figura, width="stretch", theme=None, key=chave,
                        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    @staticmethod
    def tabela(dados: pd.DataFrame, formatos: dict[str, Callable] | None = None, **opcoes) -> None:
        """
        Tabela interativa com números no padrão brasileiro (vírgula decimal).

        A formatação é feita pelo Styler do pandas: a tela mostra o texto
        formatado, mas a ordenação por coluna continua numérica.
        """
        formatos = dict(formatos or {})
        dados = dados.copy()
        # O Streamlit exibe valores ausentes como "None" mesmo com o Styler; as
        # colunas com ausências são convertidas em texto já formatado.
        for coluna in [c for c in formatos if c in dados and dados[c].isna().any()]:
            dados[coluna] = dados[coluna].map(formatos.pop(coluna))
        estilo = dados.style.format(formatos, na_rep=fmt.AUSENTE)
        st.dataframe(estilo, hide_index=True, width="stretch", **opcoes)

    def seletor_ativo(self, rotulo: str, formato_nome_primeiro: bool = True) -> str:
        """
        Caixa de seleção do ativo. A escolha é lembrada ao trocar de tela.

        Streamlit descarta o estado dos componentes que deixam de ser exibidos;
        por isso a escolha é copiada para uma variável própria da sessão.
        """
        opcoes = list(ATIVOS)
        chave_widget = f"ativo_{self.chave}"
        atual = st.session_state.get("ativo_escolhido", opcoes[0])

        def lembrar() -> None:
            st.session_state["ativo_escolhido"] = st.session_state[chave_widget]

        def nome(ticker: str) -> str:
            ativo = ATIVOS[ticker]
            return f"{ativo.nome} ({ticker})" if formato_nome_primeiro else f"{ticker} · {ativo.nome}"

        return st.selectbox(rotulo, opcoes, index=opcoes.index(atual), format_func=nome,
                            key=chave_widget, on_change=lembrar)

    def seletor_modelo(self, rotulo: str = "Modelo") -> str:
        """Caixa de seleção do modelo de classificação (escolha lembrada entre telas)."""
        disponiveis = self.ctx.modelos_disponiveis()
        opcoes = list(disponiveis) or [self.ctx.modelo_principal]
        chave_widget = f"modelo_{self.chave}"
        atual = st.session_state.get("modelo_escolhido", self.ctx.modelo_principal)
        indice = opcoes.index(atual) if atual in opcoes else 0

        def lembrar() -> None:
            st.session_state["modelo_escolhido"] = st.session_state[chave_widget]

        def nome(chave: str) -> str:
            sufixo = " · principal" if chave == self.ctx.modelo_principal else ""
            return f"{self.ctx.nome_modelo(chave)}{sufixo}"

        return st.selectbox(rotulo, opcoes, index=indice, format_func=nome, key=chave_widget, on_change=lembrar,
                            help="O modelo principal é o de maior F1-Score na validação cruzada do treinamento.")

    def seletor_periodo(self, rotulo: str = "Período do gráfico", padrao: str = "6 meses") -> int | None:
        opcoes = list(PERIODOS_GRAFICO)
        escolha = st.radio(rotulo, opcoes, index=opcoes.index(padrao), horizontal=True,
                           key=f"periodo_{self.chave}")
        return PERIODOS_GRAFICO[escolha]

    def botao_navegacao(self, rotulo: str, pagina: str, icone: str | None = None, chave: str | None = None) -> None:
        """Botão que leva a outra tela."""
        if st.button(rotulo, icon=icone, width="stretch", key=chave or f"ir_{self.chave}_{pagina}"):
            self.ctx.navegar(pagina)

    def aviso_sem_modelo(self) -> bool:
        """Mostra instruções quando não há modelos treinados; devolve True nesse caso."""
        if self.ctx.servico_sinais.treinado and self.ctx.modelos_disponiveis():
            return False
        st.warning("Os modelos ainda não foram treinados. No terminal, dentro da pasta do projeto, execute "
                   "`python treinar_modelos.py` e depois recarregue a página.")
        return True
