"""
Manchetes recentes sobre os bancos monitorados (Google Notícias, RSS).

O feed RSS é público e não requer chave. Cada item traz título, link, data de
publicação e veículo. As manchetes são apenas informativas: não entram nos
modelos de classificação.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from econom_ia.integracoes.http import ClienteHTTP, ErroIntegracao

# Horário de Brasília (UTC-3; o Brasil não adota horário de verão desde 2019).
FUSO_BRASILIA = timezone(timedelta(hours=-3))


@dataclass
class Noticia:
    titulo: str
    link: str
    fonte: str
    publicada_em: datetime | None
    ticker: str


@dataclass
class Manchetes:
    """Resultado de uma consulta: as notícias e a procedência dos dados."""

    noticias: list[Noticia]
    origem: str | None         # "online", "cache", "cache_antigo" ou None (nada obtido)
    obtido_em: datetime | None


class ClienteNoticias(ClienteHTTP):
    nome = "noticias"
    URL = "https://news.google.com/rss/search"

    # Para nomes muito comuns (Bradesco, Banco do Brasil), a busca exige termos
    # do mercado financeiro, evitando resultados como venda de ingressos.
    CONSULTAS = {
        "ITUB4": '"Itaú Unibanco" OR ITUB4',
        "BBDC4": "Bradesco (ações OR BBDC4 OR lucro OR resultado OR dividendos)",
        "BBAS3": '"Banco do Brasil" (ações OR BBAS3 OR lucro OR resultado OR dividendos)',
        "SANB11": '"Santander Brasil" OR SANB11',
    }

    # Páginas de cotação e cadastros de fundos aparecem no feed, mas não são notícias.
    NAO_NOTICIAS = re.compile(
        r"cota[çc][ãa]o hoje|pre[çc]o da a[çc][ãa]o|dividendos e indicadores|cota[çc][ãa]o e indicadores"
        r"|resp(onsabilidade)? limitada|pr[ée]-venda|ingressos",
        re.IGNORECASE,
    )
    MINIMO_PALAVRAS = 5

    def manchetes(self, por_ativo: int = 4, limite: int = 8) -> Manchetes:
        """Manchetes mais recentes (últimos 7 dias) dos quatro bancos."""
        noticias, respostas = [], []
        for ticker, consulta in self.CONSULTAS.items():
            params = {"q": f"{consulta} when:7d", "hl": "pt-BR", "gl": "BR", "ceid": "BR:pt-419"}
            try:
                resposta = self.obter_texto(self.URL, params)
            except ErroIntegracao:
                continue
            respostas.append(resposta)
            validas = [n for n in self.interpretar_rss(resposta.conteudo, ticker) if self.eh_noticia(n.titulo)]
            noticias += validas[:por_ativo]

        vistas, unicas = set(), []
        for noticia in sorted(noticias, key=lambda n: n.publicada_em or datetime.min, reverse=True):
            if noticia.titulo.lower() not in vistas:
                vistas.add(noticia.titulo.lower())
                unicas.append(noticia)
        if not respostas:
            return Manchetes([], None, None)
        # A procedência exibida é a da consulta mais antiga entre as quatro.
        mais_antiga = min(respostas, key=lambda r: r.obtido_em)
        origem = "online" if all(r.origem == "online" for r in respostas) else mais_antiga.origem
        return Manchetes(unicas[:limite], origem, mais_antiga.obtido_em)

    @classmethod
    def eh_noticia(cls, titulo: str) -> bool:
        """Descarta páginas de cotação e títulos curtos demais para serem manchetes."""
        return len(titulo.split()) >= cls.MINIMO_PALAVRAS and not cls.NAO_NOTICIAS.search(titulo)

    @staticmethod
    def interpretar_rss(texto_xml: str, ticker: str) -> list[Noticia]:
        try:
            raiz = ET.fromstring(texto_xml)
        except ET.ParseError:
            return []
        noticias = []
        for item in raiz.iter("item"):
            titulo = (item.findtext("title") or "").strip()
            fonte = (item.findtext("source") or "").strip()
            # O Google acrescenta " - Veículo" ao fim do título.
            if fonte and titulo.endswith(f" - {fonte}"):
                titulo = titulo[: -len(fonte) - 3]
            titulo = re.sub(r"^Not[íi]cia\s+", "", titulo)  # prefixo usado por alguns agregadores
            try:
                # O feed informa o horário em GMT; convertemos para Brasília.
                publicada = parsedate_to_datetime(item.findtext("pubDate") or "")
                if publicada.tzinfo is not None:
                    publicada = publicada.astimezone(FUSO_BRASILIA)
                publicada = publicada.replace(tzinfo=None)
            except (TypeError, ValueError):
                publicada = None
            if titulo:
                noticias.append(Noticia(titulo, (item.findtext("link") or "").strip(), fonte, publicada, ticker))
        return noticias
