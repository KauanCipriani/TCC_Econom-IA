"""
FRED (Federal Reserve Economic Data), do Federal Reserve Bank of St. Louis.

Usa o endereço público de exportação em CSV, que não exige chave de acesso.
A primeira coluna é a data ("observation_date" ou "DATE", conforme a versão
do serviço) e a segunda o valor; dias sem dado aparecem vazios ou com ".".
"""

from __future__ import annotations

import io

import pandas as pd

from econom_ia.integracoes.http import ClienteHTTP, SerieTemporal


class ClienteFRED(ClienteHTTP):
    nome = "fred"
    IGNORAR_NA_CHAVE = ("cosd",)
    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    SERIES = {
        "fed_funds": "DFF",          # Taxa efetiva dos Fed Funds (% a.a.)
        "treasury_3m": "DGS3MO",     # Título do Tesouro americano de 3 meses
        "treasury_2a": "DGS2",       # Título de 2 anos
        "treasury_10a": "DGS10",     # Título de 10 anos
    }

    def serie(self, chave: str, anos: int = 5) -> SerieTemporal:
        inicio = pd.Timestamp.today().normalize() - pd.DateOffset(years=anos)
        resposta = self.obter_texto(self.URL, {"id": self.SERIES[chave], "cosd": inicio.strftime("%Y-%m-%d")})
        return SerieTemporal(chave, self.interpretar(resposta.conteudo), resposta.origem, resposta.obtido_em)

    @staticmethod
    def interpretar(texto_csv: str) -> pd.DataFrame:
        bruto = pd.read_csv(io.StringIO(texto_csv))
        dados = pd.DataFrame({
            "data": pd.to_datetime(bruto.iloc[:, 0], errors="coerce"),
            "valor": pd.to_numeric(bruto.iloc[:, 1], errors="coerce"),
        })
        return dados.dropna().sort_values("data").reset_index(drop=True)
