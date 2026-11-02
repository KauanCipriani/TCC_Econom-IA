"""
Expectativas de mercado para a Selic (Relatório Focus, Banco Central do Brasil).

Toda semana o Banco Central consulta cerca de 140 instituições financeiras
sobre as suas projeções para a economia e publica o Relatório Focus. A API
Olinda (padrão OData, pública e sem chave) disponibiliza, para cada reunião
futura do Copom, a mediana, a média, o desvio-padrão e os valores mínimo e
máximo das projeções para a meta Selic.

Usamos a base de cálculo 0, que considera as projeções informadas nos
últimos 30 dias (a mesma do relatório semanal). As reuniões vêm codificadas
como "R7/2026" (sétima reunião de 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from econom_ia.integracoes.http import ClienteHTTP


@dataclass
class ExpectativasSelic:
    """Projeções do mercado para a meta Selic nas próximas reuniões do Copom."""

    data_referencia: pd.Timestamp   # data da pesquisa Focus
    reunioes: pd.DataFrame          # reuniao, rotulo, mediana, media, desvio, minimo, maximo, respondentes
    origem: str
    obtido_em: datetime

    @property
    def proxima(self) -> pd.Series | None:
        """Expectativa para a próxima reunião do Copom."""
        return self.reunioes.iloc[0] if len(self.reunioes) else None

    def fim_de_ano(self, ano: int) -> pd.Series | None:
        """Expectativa para a última reunião do ano (Selic ao fim do ano)."""
        do_ano = self.reunioes[self.reunioes["ano"] == ano]
        return do_ano.iloc[-1] if len(do_ano) else None


class ClienteFocus(ClienteHTTP):
    nome = "focus"
    URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoSelic"
    # Parâmetros OData já codificados (o servidor não aceita "+" no lugar de espaço).
    CONSULTA = ("$top=60&$orderby=Data%20desc&$filter=baseCalculo%20eq%200&$format=json"
                "&$select=Data,Reuniao,Media,Mediana,DesvioPadrao,Minimo,Maximo,numeroRespondentes")

    def expectativas_selic(self) -> ExpectativasSelic:
        resposta = self.obter_json(f"{self.URL}?{self.CONSULTA}")
        data, reunioes = self.interpretar(resposta.conteudo)
        return ExpectativasSelic(data, reunioes, resposta.origem, resposta.obtido_em)

    @staticmethod
    def interpretar(conteudo: dict) -> tuple[pd.Timestamp, pd.DataFrame]:
        """
        Mantém apenas a pesquisa mais recente e ordena as reuniões no tempo.

        Raises:
            ValueError: resposta sem projeções.
        """
        valores = pd.DataFrame((conteudo or {}).get("value", []))
        if valores.empty:
            raise ValueError("Resposta do Focus sem projeções.")
        valores["Data"] = pd.to_datetime(valores["Data"])
        recente = valores[valores["Data"] == valores["Data"].max()].copy()
        partes = recente["Reuniao"].str.extract(r"R(\d+)/(\d{4})").astype(int)
        recente["numero"], recente["ano"] = partes[0], partes[1]
        recente = recente.sort_values(["ano", "numero"])
        reunioes = pd.DataFrame({
            "reuniao": recente["Reuniao"],
            "rotulo": [f"{n}ª reunião de {a}" for n, a in zip(recente["numero"], recente["ano"], strict=True)],
            "ano": recente["ano"],
            "mediana": recente["Mediana"].astype(float),
            "media": recente["Media"].astype(float),
            "desvio": recente["DesvioPadrao"].astype(float),
            "minimo": recente["Minimo"].astype(float),
            "maximo": recente["Maximo"].astype(float),
            "respondentes": recente["numeroRespondentes"].astype(int),
        }).reset_index(drop=True)
        return recente["Data"].iloc[0], reunioes
