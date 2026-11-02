"""
Serviços consumidos pela interface: cotações, sinais, previsões e alertas.

Cada serviço encapsula uma regra de negócio e esconde da interface os
detalhes de onde vêm os dados e de como os modelos foram treinados.
"""

from econom_ia.servicos.alertas import CentralAlertas
from econom_ia.servicos.cotacoes import CotacoesCarregadas, ProvedorCotacoes
from econom_ia.servicos.previsao import ServicoPrevisao
from econom_ia.servicos.sinais import ServicoSinais

__all__ = [
    "CentralAlertas",
    "CotacoesCarregadas",
    "ProvedorCotacoes",
    "ServicoPrevisao",
    "ServicoSinais",
]
