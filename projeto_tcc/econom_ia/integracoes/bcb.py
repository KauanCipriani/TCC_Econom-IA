"""
API de Séries Temporais do Banco Central do Brasil (SGS).

Formato da resposta: [{"data": "dd/mm/aaaa", "valor": "13.75"}, ...].
A série da meta Selic traz valores com datas futuras (a meta vigente é
repetida até a próxima reunião do Copom); por isso descartamos datas
posteriores a hoje.
"""

from __future__ import annotations

import pandas as pd

from econom_ia.integracoes.http import ClienteHTTP, SerieTemporal


class ClienteBCB(ClienteHTTP):
    nome = "bcb"
    IGNORAR_NA_CHAVE = ("dataInicial", "dataFinal")
    URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"

    # Códigos das séries no SGS.
    SERIES = {
        "selic_meta": 432,      # Meta da taxa Selic definida pelo Copom (% a.a.)
        "ipca_12m": 13522,      # IPCA acumulado em 12 meses (%)
        "dolar_ptax": 1,        # Dólar comercial, venda, PTAX (R$)
    }

    def serie(self, chave: str, anos: int = 5) -> SerieTemporal:
        """Série dos últimos "anos" anos, com as colunas data e valor."""
        hoje = pd.Timestamp.today().normalize()
        params = {
            "formato": "json",
            "dataInicial": (hoje - pd.DateOffset(years=anos)).strftime("%d/%m/%Y"),
            "dataFinal": hoje.strftime("%d/%m/%Y"),
        }
        resposta = self.obter_json(self.URL.format(codigo=self.SERIES[chave]), params)
        return SerieTemporal(chave, self.interpretar(resposta.conteudo, hoje), resposta.origem, resposta.obtido_em)

    @staticmethod
    def interpretar(conteudo: list[dict], ate: pd.Timestamp | None = None) -> pd.DataFrame:
        """Converte a lista do SGS em DataFrame numérico e ordenado."""
        dados = pd.DataFrame(conteudo, columns=["data", "valor"])
        dados["data"] = pd.to_datetime(dados["data"], format="%d/%m/%Y", errors="coerce")
        dados["valor"] = pd.to_numeric(dados["valor"], errors="coerce")
        dados = dados.dropna()
        if ate is not None:
            dados = dados[dados["data"] <= ate]
        return dados.sort_values("data").reset_index(drop=True)
