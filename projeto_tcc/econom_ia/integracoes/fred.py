"""
FRED (Federal Reserve Economic Data), do Federal Reserve Bank of St. Louis.

Há dois caminhos:
  * com chave de acesso (gratuita, obtida em fredaccount.stlouisfed.org), usa a
    API oficial em JSON, recomendada para servidores na internet;
  * sem chave, usa o endereço público de exportação em CSV, suficiente no
    computador pessoal, mas que costuma recusar acessos vindos de servidores
    de nuvem.

A chave é lida do argumento chave_api ou da variável de ambiente
FRED_API_KEY; na interface publicada, vem dos "secrets" do Streamlit.
No CSV, a primeira coluna é a data ("observation_date" ou "DATE", conforme a
versão do serviço) e a segunda o valor; dias sem dado aparecem vazios ou com ".".
"""

from __future__ import annotations

import io
import os

import pandas as pd

from econom_ia.integracoes.http import ClienteHTTP, SerieTemporal


class ClienteFRED(ClienteHTTP):
    nome = "fred"
    # As datas mudam todo dia e a chave não identifica o dado: ficam fora do nome da cópia local.
    IGNORAR_NA_CHAVE = ("cosd", "observation_start", "api_key")
    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    URL_API = "https://api.stlouisfed.org/fred/series/observations"

    SERIES = {
        "fed_funds": "DFF",          # Taxa efetiva dos Fed Funds (% a.a.)
        "treasury_3m": "DGS3MO",     # Título do Tesouro americano de 3 meses
        "treasury_2a": "DGS2",       # Título de 2 anos
        "treasury_10a": "DGS10",     # Título de 10 anos
    }

    def __init__(self, *args, chave_api: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.chave_api = chave_api or os.environ.get("FRED_API_KEY") or None

    def serie(self, chave: str, anos: int = 5) -> SerieTemporal:
        inicio = (pd.Timestamp.today().normalize() - pd.DateOffset(years=anos)).strftime("%Y-%m-%d")
        codigo = self.SERIES[chave]
        if self.chave_api:
            resposta = self.obter_json(self.URL_API, {"series_id": codigo, "file_type": "json",
                                                      "observation_start": inicio, "api_key": self.chave_api})
            dados = self.interpretar_json(resposta.conteudo)
        else:
            resposta = self.obter_texto(self.URL, {"id": codigo, "cosd": inicio})
            dados = self.interpretar(resposta.conteudo)
        return SerieTemporal(chave, dados, resposta.origem, resposta.obtido_em)

    @staticmethod
    def interpretar(texto_csv: str) -> pd.DataFrame:
        bruto = pd.read_csv(io.StringIO(texto_csv))
        dados = pd.DataFrame({
            "data": pd.to_datetime(bruto.iloc[:, 0], errors="coerce"),
            "valor": pd.to_numeric(bruto.iloc[:, 1], errors="coerce"),
        })
        return dados.dropna().sort_values("data").reset_index(drop=True)

    @staticmethod
    def interpretar_json(conteudo: dict) -> pd.DataFrame:
        """Resposta da API oficial: {"observations": [{"date": "2026-10-01", "value": "4.33"}, ...]}."""
        observacoes = conteudo["observations"]
        dados = pd.DataFrame({
            "data": pd.to_datetime([o.get("date") for o in observacoes], errors="coerce"),
            "valor": pd.to_numeric([o.get("value") for o in observacoes], errors="coerce"),
        })
        return dados.dropna().sort_values("data").reset_index(drop=True)
