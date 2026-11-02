"""Formatação de números e datas no padrão brasileiro."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime

import pandas as pd

DIAS_SEMANA = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira",
               "Sexta-feira", "Sábado", "Domingo"]
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
         "agosto", "setembro", "outubro", "novembro", "dezembro"]
AUSENTE = "—"


def ausente(valor) -> bool:
    """Verdadeiro para None, NaN, NaT e pd.NA."""
    try:
        return valor is None or bool(pd.isna(valor))
    except (TypeError, ValueError):  # listas e outros objetos não escalares
        return False


def numero(valor: float | None, casas: int = 2) -> str:
    """1234.5 -> '1.234,50'."""
    if ausente(valor):
        return AUSENTE
    texto = f"{float(valor):,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def moeda(valor: float | None, casas: int = 2) -> str:
    """12.3 -> 'R$ 12,30'."""
    return AUSENTE if ausente(valor) else f"R$ {numero(valor, casas)}"


def percentual(valor: float | None, casas: int = 2, sinal: bool = True) -> str:
    """Recebe o valor já em pontos percentuais (1.5 -> '+1,50%')."""
    if ausente(valor):
        return AUSENTE
    prefixo = "+" if sinal and valor > 0 else ""
    return f"{prefixo}{numero(valor, casas)}%"


def pontos_percentuais(valor: float | None, casas: int = 2) -> str:
    """Diferença entre taxas: 0.25 -> '+0,25 p.p.'."""
    if ausente(valor):
        return AUSENTE
    return f"{'+' if valor > 0 else ''}{numero(valor, casas)} p.p."


def numero_compacto(valor: float | None) -> str:
    """Volumes grandes: 12_345_678 -> '12,3 mi'."""
    if ausente(valor):
        return AUSENTE
    for divisor, sufixo in ((1e9, "bi"), (1e6, "mi"), (1e3, "mil")):
        if abs(valor) >= divisor:
            return f"{numero(valor / divisor, 1)} {sufixo}"
    return numero(valor, 0)


def formatador(casas: int = 2, sinal: bool = False, sufixo: str = "") -> Callable[[float], str]:
    """Função de formatação para colunas de tabelas (ex.: formatador(1, sufixo="%"))."""
    def aplicar(valor) -> str:
        if ausente(valor):
            return AUSENTE
        texto = numero(valor, casas)
        return f"{'+' if sinal and valor > 0 else ''}{texto}{sufixo}"
    return aplicar


def escapar_markdown(texto: str) -> str:
    """Evita que o "$" de "R$" seja lido como fórmula matemática pelo Markdown."""
    return texto.replace("$", "\\$")


def data_curta(valor) -> str:
    """Data no formato dd/mm/aaaa."""
    return AUSENTE if ausente(valor) else f"{valor:%d/%m/%Y}"


def mes_ano(valor) -> str:
    """'set/2026'."""
    return AUSENTE if ausente(valor) else f"{MESES[valor.month - 1][:3]}/{valor.year}"


def data_hora(valor: datetime | None) -> str:
    """'28/09 às 18:40'."""
    return AUSENTE if ausente(valor) else f"{valor:%d/%m} às {valor:%H:%M}"


def data_extenso(valor: date | datetime) -> str:
    """'Segunda-feira, 28 de setembro de 2026'."""
    return f"{DIAS_SEMANA[valor.weekday()]}, {valor.day} de {MESES[valor.month - 1]} de {valor.year}"
