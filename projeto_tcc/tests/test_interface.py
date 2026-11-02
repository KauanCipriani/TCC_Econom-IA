"""
Testes de fumaça da interface (Streamlit AppTest).

Executam o aplicativo e cada tela com os modelos treinados e a base local do
projeto, sem internet, e verificam que nenhuma exceção é lançada. São
ignorados se o treinamento ainda não foi executado.
"""

import pytest

from econom_ia.config import CAMINHOS

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(CAMINHOS.raiz / "app.py")
TELAS = ["visao_geral", "painel_iniciante", "painel_avancado", "previsoes_macro", "centro_alertas", "desempenho"]

pytestmark = pytest.mark.skipif(not CAMINHOS.metricas.exists() or not CAMINHOS.ohlcv.exists(),
                                reason="execute python treinar_modelos.py antes dos testes da interface")


def _tela_do_estado() -> None:
    """Abre a tela indicada em st.session_state["tela_teste"] (sem o menu lateral)."""
    import streamlit as st

    from econom_ia.interface.aplicacao import AplicacaoEconomIA

    AplicacaoEconomIA().executar_tela(st.session_state["tela_teste"])


def _iniciar(tela: str) -> AppTest:
    app = AppTest.from_function(_tela_do_estado, default_timeout=300)
    app.session_state["tela_teste"] = tela
    app.session_state["usar_internet"] = False
    return app.run()


def test_aplicativo_abre_com_menu_lateral():
    app = AppTest.from_file(APP, default_timeout=300)
    app.session_state["usar_internet"] = False
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.toggle(key="usar_internet").value is False  # controles na barra lateral


@pytest.mark.parametrize("tela", TELAS)
def test_tela_abre_sem_erros(tela):
    app = _iniciar(tela)
    assert not app.exception, [e.value for e in app.exception]


def test_troca_de_modelo_lembrada_entre_telas():
    app = _iniciar("painel_avancado")
    for modelo in ("random_forest", "svm", "lstm"):
        app.selectbox(key="modelo_painel_avancado").select(modelo).run()
        assert not app.exception, [e.value for e in app.exception]
    app.session_state["tela_teste"] = "centro_alertas"
    app.run()
    assert app.selectbox(key="modelo_centro_alertas").value == "lstm"  # escolha lembrada entre telas
    assert not app.exception


def test_personalizar_esconde_e_mostra_blocos():
    app = _iniciar("painel_avancado")
    titulos = lambda: " ".join(m.value for m in app.markdown)  # noqa: E731
    assert 'titulo-secao">Indicadores técnicos' in titulos()
    app.toggle(key="mostrar_painel_avancado_indicadores").set_value(False).run()
    assert 'titulo-secao">Indicadores técnicos' not in titulos()
    app.session_state["tela_teste"] = "centro_alertas"
    app.run()
    app.session_state["tela_teste"] = "painel_avancado"
    app.run()
    assert 'titulo-secao">Indicadores técnicos' not in titulos()  # escolha guardada ao trocar de tela
    app.button(key="mostrar_tudo_painel_avancado").click().run()
    assert 'titulo-secao">Indicadores técnicos' in titulos()
    assert not app.exception, [e.value for e in app.exception]
