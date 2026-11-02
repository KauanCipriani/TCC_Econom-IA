"""
Calendário de pregões da B3 usado para datar as previsões.

Lista os feriados em que não há negociação na B3 (nacionais, Carnaval,
Sexta-feira Santa, Corpus Christi, véspera de Natal e último dia do ano).
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

FERIADOS_B3 = pd.to_datetime([
    # 2025
    "2025-01-01", "2025-03-03", "2025-03-04", "2025-04-18", "2025-04-21", "2025-05-01",
    "2025-06-19", "2025-11-20", "2025-12-24", "2025-12-25", "2025-12-31",
    # 2026
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-04-03", "2026-04-21", "2026-05-01",
    "2026-06-04", "2026-09-07", "2026-10-12", "2026-11-02", "2026-11-20", "2026-12-24",
    "2026-12-25", "2026-12-31",
    # 2027
    "2027-01-01", "2027-02-08", "2027-02-09", "2027-03-26", "2027-04-21", "2027-05-27",
    "2027-09-07", "2027-10-12", "2027-11-02", "2027-11-15", "2027-12-24", "2027-12-31",
])

DIA_DE_PREGAO = CustomBusinessDay(holidays=FERIADOS_B3)


def proximos_pregoes(ultima_data: pd.Timestamp, quantidade: int) -> pd.DatetimeIndex:
    """Datas dos "quantidade" pregões seguintes a "ultima_data"."""
    return pd.date_range(pd.Timestamp(ultima_data) + DIA_DE_PREGAO, periods=quantidade, freq=DIA_DE_PREGAO)
