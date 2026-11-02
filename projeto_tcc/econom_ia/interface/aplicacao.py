"""
Aplicação Streamlit: configuração da página, menu lateral e roteamento.

A cada interação o Streamlit executa o script inteiro de novo. A classe
AplicacaoEconomIA organiza essa execução: aplica o tema, registra as telas no
menu da barra lateral (st.navigation), monta o contexto (cotações, modelos e
serviços, todos em cache), entrega o controle à tela escolhida, que é uma
subclasse de Tela, e mostra na barra lateral a situação do sistema.

A barra começa aberta e pode ser recolhida pelo botão «; recolhida, o botão
"Menu" a reabre, então nunca se fica preso em uma tela.

Cada tela tem um endereço próprio (por exemplo, /painel_avancado), então o
botão "voltar" do navegador funciona e uma tela pode ser aberta pelo link.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from econom_ia.interface import dados_app
from econom_ia.interface import formatacao as fmt
from econom_ia.interface.componentes import Componentes
from econom_ia.interface.contexto import ContextoApp
from econom_ia.interface.telas import TELAS, Tela
from econom_ia.interface.tema import LOGO_MARCA_SVG, Paleta, aplicar_tema


class AplicacaoEconomIA:
    """Ponto de entrada da interface web."""

    PAGINA_INICIAL = "visao_geral"
    # Grupos do menu lateral e as telas de cada um, na ordem de exibição.
    SECOES = {
        "Início": ("visao_geral",),
        "Análise": ("painel_iniciante", "painel_avancado"),
        "Ferramentas": ("previsoes_macro", "centro_alertas", "desempenho"),
    }

    def __init__(self, telas: tuple[type[Tela], ...] = TELAS) -> None:
        self.telas = {tela.chave: tela for tela in telas}
        self.contexto: ContextoApp | None = None

    def executar(self) -> None:
        """Execução normal: menu lateral com todas as telas."""
        self._preparar()
        paginas = {
            chave: st.Page(self._executor(tela), title=tela.rotulo, icon=tela.icone, url_path=chave,
                           default=chave == self.PAGINA_INICIAL)
            for chave, tela in self.telas.items()
        }
        ContextoApp.paginas = paginas
        grupos = {secao: [paginas[c] for c in chaves if c in paginas] for secao, chaves in self.SECOES.items()}
        navegacao = st.navigation(grupos, position="sidebar")
        self.contexto = self._criar_contexto()
        with st.sidebar:
            self._situacao(self.contexto)
        navegacao.run()

    def executar_tela(self, chave: str) -> None:
        """Abre uma única tela, sem o menu (usado pelos testes automatizados)."""
        self._preparar()
        self.contexto = self._criar_contexto()
        self._executor(self.telas[chave])()
        with st.sidebar:
            self._situacao(self.contexto)

    # ── Preparação e contexto ──────────────────────────────────────────────
    @staticmethod
    def _preparar() -> None:
        st.set_page_config(page_title="Econom-IA", page_icon=":material/insights:", layout="wide",
                           initial_sidebar_state="expanded")
        aplicar_tema()
        st.logo(LOGO_MARCA_SVG, size="large")
        st.session_state.setdefault("usar_internet", True)

    def _executor(self, tela: type[Tela]) -> Callable[[], None]:
        """Função que desenha a tela (o st.Page recebe uma função sem argumentos)."""
        def desenhar() -> None:
            if self.contexto is not None:
                tela(self.contexto).renderizar()
        desenhar.__name__ = f"tela_{tela.chave}"
        return desenhar

    @staticmethod
    def _criar_contexto() -> ContextoApp | None:
        usar_internet = bool(st.session_state["usar_internet"])
        try:
            cotacoes = dados_app.cotacoes(usar_internet)
        except FileNotFoundError:
            st.error("Nenhuma base de cotações foi encontrada. No terminal, dentro da pasta do projeto, execute "
                     "`python coletar_dados.py` (requer internet) e depois `python treinar_modelos.py`.")
            return None
        return ContextoApp(cotacoes, dados_app.servico_sinais(), dados_app.servico_previsao(), usar_internet)

    # ── Situação do sistema (barra lateral) ────────────────────────────────
    @staticmethod
    def _situacao(contexto: ContextoApp | None) -> None:
        """Situação dos modelos e dos dados, controles de atualização e créditos."""
        Componentes.renderizar('<div class="secao-lateral">Situação do sistema</div>')
        if contexto is not None:
            metricas = contexto.metricas
            if metricas:
                principal = metricas["modelo_principal"]
                f1 = metricas["modelos"][principal]["teste"]["f1_macro"]
                data = metricas["info"]["data_treinamento"][:10]
                Componentes.renderizar(
                    '<div class="status"><div class="status-titulo" style="color:#34d399;">✔ Modelos treinados'
                    f'</div><div class="status-texto">Principal: {contexto.nome_modelo(principal)}<br>F1-Score no '
                    f'teste: {fmt.numero(f1, 3)}<br>Treinados em {data[8:10]}/{data[5:7]}/{data[:4]}</div></div>')
            else:
                Componentes.renderizar(
                    f'<div class="status"><div class="status-titulo" style="color:{Paleta.ALERTA};">'
                    '⚠ Modelos não treinados</div><div class="status-texto">No terminal, execute '
                    '<code>python treinar_modelos.py</code></div></div>')
            cotacoes = contexto.cotacoes
            origem = "Cotações ao vivo" if cotacoes.origem == "online" else "Cotações da base local"
            Componentes.renderizar(
                f'<div class="status"><div class="status-titulo" style="color:#8fc0f5;">● {origem}</div>'
                f'<div class="status-texto">Último pregão: {fmt.data_curta(cotacoes.ultima_data)}<br>'
                f'Carregadas em {fmt.data_hora(cotacoes.carregado_em)}<br>Renovação a cada 15 minutos</div></div>')
        st.toggle("Buscar dados na internet", key="usar_internet",
                  help="Desligado, o sistema usa apenas a base local de cotações e as últimas cópias salvas das "
                       "APIs (útil sem conexão).")
        st.button("Atualizar dados agora", icon=":material/refresh:", width="stretch", key="atualizar",
                  on_click=dados_app.limpar_cache)
        Componentes.renderizar(
            '<div class="creditos"><strong>TCC · Ciência da Computação</strong><br>Unoesc São Miguel do Oeste, '
            '2026<br>Kauan Amélio Cipriani<br>Vitor Hugo Konzen<br>Orientador: Vinícius Almeida dos Santos'
            '<span class="aviso-lateral">Ferramenta acadêmica. Não é recomendação de investimento.</span></div>')
