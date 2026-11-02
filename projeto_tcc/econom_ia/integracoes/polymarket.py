"""
Mercado de previsão Polymarket: probabilidades para a próxima decisão do Copom.

O Polymarket publica, antes de cada reunião do Copom, um evento chamado
"Bank of Brazil decision in <mês>?", com um mercado por resultado possível
(manutenção, corte, alta). O preço do "Sim" de cada mercado, entre 0 e 1,
é interpretado como a probabilidade coletiva daquele resultado.

A consulta usa a Gamma API pública (sem chave). Primeiro tentamos os
endereços (slugs) dos próximos meses; se nada for encontrado, recorremos à
busca textual. Se não houver evento ativo, a tela informa isso ao usuário.

Observação: desde abril de 2026 o acesso ao Polymarket é bloqueado no Brasil
por determinação da Secretaria de Prêmios e Apostas do Ministério da Fazenda.
Em redes brasileiras a consulta pode falhar; nesse caso a busca é
interrompida na primeira falha de conexão (para não travar a interface) e a
tela exibe apenas as expectativas do Relatório Focus.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from econom_ia.integracoes.http import ClienteHTTP

MESES_EN = ["january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december"]


@dataclass
class MercadoPrevisao:
    """Evento do Polymarket já convertido para exibição."""

    titulo: str
    url: str
    data_encerramento: pd.Timestamp | None
    volume: float | None
    opcoes: list[tuple[str, float]] = field(default_factory=list)  # (resultado, probabilidade)
    origem: str = "online"
    obtido_em: datetime | None = None

    @property
    def mais_provavel(self) -> tuple[str, float] | None:
        return max(self.opcoes, key=lambda opcao: opcao[1]) if self.opcoes else None


class ClientePolymarket(ClienteHTTP):
    nome = "polymarket"
    TENTATIVAS = 1  # bloqueado no Brasil: repetir só atrasaria a tela
    TEMPO_LIMITE_S = 6.0
    URL_EVENTOS = "https://gamma-api.polymarket.com/events"
    URL_BUSCA = "https://gamma-api.polymarket.com/public-search"
    URL_SITE = "https://polymarket.com/event/{slug}"

    def proxima_decisao_copom(self, hoje: pd.Timestamp | None = None) -> MercadoPrevisao | None:
        """
        Evento ativo mais próximo sobre a decisão da Selic, ou None se não houver.

        Raises:
            ErroIntegracao: a Gamma API não respondeu (sem internet ou acesso
                bloqueado) e não há cópia local da consulta.
        """
        hoje = (hoje or pd.Timestamp.today()).normalize()
        candidatos = []
        for slug in self._slugs_candidatos(hoje):
            # Uma falha de conexão interrompe a busca: se a API não respondeu
            # para um endereço, também não responderá para os demais.
            resposta = self.obter_json(self.URL_EVENTOS, {"slug": slug})
            candidatos += [(evento, resposta) for evento in (resposta.conteudo or [])]
        if not any(self.eh_valido(evento, hoje) for evento, _ in candidatos):
            resposta = self.obter_json(self.URL_BUSCA, {"q": "Bank of Brazil decision", "events_status": "active"})
            candidatos += [(evento, resposta) for evento in (resposta.conteudo or {}).get("events", [])]

        validos = [(_data_encerramento(evento) or pd.Timestamp.max, evento, resposta)
                   for evento, resposta in candidatos if self.eh_valido(evento, hoje)]
        if not validos:
            return None
        _, evento, resposta = min(validos, key=lambda item: item[0])
        return self.converter(evento, resposta.origem, resposta.obtido_em)

    @classmethod
    def eh_valido(cls, evento: dict, hoje: pd.Timestamp) -> bool:
        """Evento sobre o Copom, aberto a negociações e ainda não encerrado."""
        fim = _data_encerramento(evento)
        return (cls.eh_evento_copom(evento) and not evento.get("closed") and not evento.get("archived")
                and evento.get("active", True) is not False and (fim is None or fim >= hoje))

    @staticmethod
    def _slugs_candidatos(hoje: pd.Timestamp) -> list[str]:
        slugs = []
        for deslocamento in range(0, 4):
            data = hoje + pd.DateOffset(months=deslocamento)
            mes = MESES_EN[data.month - 1]
            slugs += [f"bank-of-brazil-decision-in-{mes}-{data.year}", f"bank-of-brazil-decision-in-{mes}"]
        return slugs

    @staticmethod
    def eh_evento_copom(evento: dict) -> bool:
        titulo = (evento.get("title") or "").lower()
        return "brazil" in titulo and any(p in titulo for p in ("decision", "selic", "interest rate"))

    @classmethod
    def converter(cls, evento: dict, origem: str = "online", obtido_em: datetime | None = None) -> MercadoPrevisao:
        opcoes = []
        for mercado in evento.get("markets", []):
            if mercado.get("active") is False:
                continue
            resultados = _lista(mercado.get("outcomes"))
            precos = [float(p) for p in _lista(mercado.get("outcomePrices"))]
            if not resultados or len(resultados) != len(precos):
                continue
            if [r.lower() for r in resultados] == ["yes", "no"]:
                rotulo = mercado.get("groupItemTitle") or mercado.get("question", "")
                opcoes.append((cls.traduzir(rotulo), precos[0]))
            else:
                opcoes += [(cls.traduzir(r), p) for r, p in zip(resultados, precos, strict=True)]
        opcoes.sort(key=lambda opcao: -opcao[1])
        volume = evento.get("volume")
        return MercadoPrevisao(
            titulo=evento.get("title", "Decisão do Copom"),
            url=cls.URL_SITE.format(slug=evento.get("slug", "")),
            data_encerramento=_data_encerramento(evento),
            volume=float(volume) if volume not in (None, "") else None,
            opcoes=opcoes,
            origem=origem,
            obtido_em=obtido_em,
        )

    @staticmethod
    def traduzir(rotulo: str) -> str:
        """Traduz os resultados mais comuns (ex.: '25 bps decrease')."""
        texto = rotulo.strip()
        minusculo = texto.lower()
        movimento = re.search(r"(\d+)\s*(\+?)\s*bps?\s*(decrease|cut|increase|hike)", minusculo)
        if movimento:
            pontos = int(movimento.group(1)) / 100
            acao = "Corte" if movimento.group(3) in ("decrease", "cut") else "Alta"
            sufixo = " ou mais" if movimento.group(2) else ""
            return f"{acao} de {pontos:.2f} p.p.{sufixo}".replace(".", ",", 1)
        if "no change" in minusculo or "unchanged" in minusculo or "hold" in minusculo:
            return "Manutenção da Selic"
        if minusculo in ("decrease", "cut", "lower"):
            return "Corte da Selic"
        if minusculo in ("increase", "hike", "raise"):
            return "Alta da Selic"
        return texto


def _data_encerramento(evento: dict) -> pd.Timestamp | None:
    fim = pd.to_datetime(evento.get("endDate"), errors="coerce", utc=True)
    return None if fim is None or pd.isna(fim) else fim.tz_localize(None)


def _lista(valor) -> list:
    """Na Gamma API, outcomes e outcomePrices chegam como texto JSON."""
    if isinstance(valor, str):
        try:
            return list(json.loads(valor))
        except ValueError:
            return []
    return list(valor or [])
