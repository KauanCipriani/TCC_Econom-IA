"""Telas da interface, na ordem em que aparecem no menu lateral."""

from econom_ia.interface.telas.base import Tela
from econom_ia.interface.telas.centro_alertas import TelaCentroAlertas
from econom_ia.interface.telas.desempenho import TelaDesempenho
from econom_ia.interface.telas.painel_avancado import TelaPainelAvancado
from econom_ia.interface.telas.painel_iniciante import TelaPainelIniciante
from econom_ia.interface.telas.previsoes_macro import TelaPrevisoesMacro
from econom_ia.interface.telas.visao_geral import TelaVisaoGeral

TELAS: tuple[type[Tela], ...] = (
    TelaVisaoGeral,
    TelaPainelIniciante,
    TelaPainelAvancado,
    TelaPrevisoesMacro,
    TelaCentroAlertas,
    TelaDesempenho,
)

__all__ = [
    "TELAS",
    "Tela",
    "TelaCentroAlertas",
    "TelaDesempenho",
    "TelaPainelAvancado",
    "TelaPainelIniciante",
    "TelaPrevisoesMacro",
    "TelaVisaoGeral",
]
