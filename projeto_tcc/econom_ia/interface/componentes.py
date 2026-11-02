"""
Componentes visuais reutilizáveis (HTML estilizado pelo tema).

Cada método devolve um trecho de HTML e o método renderizar() o exibe. Textos
vindos de fontes externas (manchetes, por exemplo) são escapados, para que
conteúdo de terceiros não altere a página. Os estados de compra e venda nunca
são indicados apenas pela cor: sempre há um rótulo ou um ícone junto.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st

from econom_ia.config import AVISO_CVM
from econom_ia.interface import formatacao as fmt

ROTULOS_SINAL = {"compra": "COMPRA", "venda": "VENDA", "neutro": "NEUTRO"}
ICONES_SINAL = {"compra": "▲", "venda": "▼", "neutro": "●"}


class Componentes:
    """Fábrica de componentes HTML do Econom-IA."""

    @staticmethod
    def renderizar(html: str) -> None:
        st.markdown(html, unsafe_allow_html=True)

    # ── Textos ─────────────────────────────────────────────────────────────
    @staticmethod
    def cabecalho(titulo: str, subtitulo: str = "") -> str:
        sub = f'<div class="subtitulo-pagina">{subtitulo}</div>' if subtitulo else ""
        return f'<div class="titulo-pagina">{titulo}</div>{sub}'

    @staticmethod
    def titulo_secao(texto: str, detalhe: str = "") -> str:
        extra = f' <span class="titulo-secao-detalhe">{detalhe}</span>' if detalhe else ""
        return f'<div class="titulo-secao">{texto}{extra}</div>'

    @staticmethod
    def nota(texto_html: str) -> str:
        return f'<div class="nota">{texto_html}</div>'

    @staticmethod
    def aviso_cvm() -> str:
        return f'<div class="aviso"><strong class="aviso-titulo">Aviso importante:</strong> {AVISO_CVM}</div>'

    @staticmethod
    def texto_seguro(texto: str | None) -> str:
        return escape(texto or "")

    @staticmethod
    def origem(origem: str | None, obtido_em: datetime | None = None) -> str:
        """
        Procedência de um dado externo.

        "ao vivo" (consultado agora), "atualizado às 18:40" (cópia recente,
        dentro da validade) ou "cópia local de 27/09 às 18:40" (fonte fora do ar).
        """
        if origem is None:
            return "indisponível"
        if origem == "online" or obtido_em is None:
            return "ao vivo"
        if origem == "cache":
            if obtido_em.date() == datetime.now().date():
                return f"atualizado às {obtido_em:%H:%M}"
            return f"atualizado em {fmt.data_hora(obtido_em)}"
        return f"cópia local de {fmt.data_hora(obtido_em)}"

    # ── Etiquetas e variações ──────────────────────────────────────────────
    @staticmethod
    def etiqueta_sinal(sinal: str | None) -> str:
        sinal = sinal if sinal in ROTULOS_SINAL else "neutro"
        return (f'<span class="etiqueta etiqueta-{sinal}">{ICONES_SINAL[sinal]} '
                f'{ROTULOS_SINAL[sinal]}</span>')

    @staticmethod
    def etiqueta(texto: str, tipo: str = "info") -> str:
        return f'<span class="etiqueta etiqueta-{tipo}">{escape(texto)}</span>'

    @staticmethod
    def variacao(valor_pct: float | None, sufixo: str = "", casas: int = 2) -> str:
        """Variação percentual colorida, com seta (a cor nunca vem sozinha)."""
        if fmt.ausente(valor_pct):
            return f'<span class="estavel">{fmt.AUSENTE}</span>'
        if valor_pct > 0:
            classe, seta = "sobe", "▲"
        elif valor_pct < 0:
            classe, seta = "desce", "▼"
        else:
            classe, seta = "estavel", "●"
        return f'<span class="{classe}">{seta} {fmt.percentual(valor_pct, casas)}{sufixo}</span>'

    # ── Cartões ────────────────────────────────────────────────────────────
    @staticmethod
    def cartao_valor(rotulo: str, valor: str, unidade: str = "", detalhe: str = "", fonte: str = "",
                     texto: bool = False) -> str:
        """Cartão com um valor em destaque; texto=True usa letra menor para valores escritos por extenso."""
        classe_valor = "cartao-valor cartao-valor-texto" if texto else "cartao-valor"
        unidade_html = f' <span class="cartao-unidade">{unidade}</span>' if unidade else ""
        detalhe_html = f'<div class="cartao-detalhe">{detalhe}</div>' if detalhe else ""
        fonte_html = f'<div class="cartao-fonte">{fonte}</div>' if fonte else ""
        return (f'<div class="cartao"><div class="cartao-rotulo">{rotulo}</div>'
                f'<div class="{classe_valor}">{valor}{unidade_html}</div>{detalhe_html}{fonte_html}</div>')

    @staticmethod
    def cartao_indicador(nome: str, valor: str, descricao: str, estado: str = "neutro",
                         leitura: str | None = None) -> str:
        """
        Cartão de indicador técnico com a sua leitura.

        A ordem é sempre a mesma (nome, valor, leitura e explicação), para que
        os cartões lado a lado fiquem alinhados e possam ser comparados.

        Args:
            estado: compra (leitura de alta), venda (de baixa), neutro, info ou aviso.
            leitura: texto da etiqueta; por padrão Alta, Baixa ou Neutro.
        """
        padrao = {"compra": "▲ Alta", "venda": "▼ Baixa", "neutro": "● Neutro"}
        texto = leitura or padrao.get(estado, "● Neutro")
        return (
            '<div class="cartao cartao-indicador">'
            f'<div class="indicador-nome">{nome}</div>'
            f'<div class="indicador-valor">{valor}</div>'
            f'<div style="margin-bottom:10px;"><span class="etiqueta etiqueta-{estado}">{texto}</span></div>'
            f'<div class="indicador-desc">{descricao}</div></div>'
        )

    @classmethod
    def cartao_banco(cls, ticker: str, nome: str, preco: str, variacao_html: str, sinal: str | None,
                     confianca: str = "") -> str:
        """Cartão com o sinal do dia de um banco (Visão Geral)."""
        classe = sinal if sinal in ROTULOS_SINAL else "neutro"
        confianca_html = f'<div class="banco-confianca">{confianca}</div>' if confianca else ""
        return (f'<div class="banco banco-{classe}"><div class="banco-ticker">{ticker}</div>'
                f'<div class="banco-nome">{nome}</div>{cls.etiqueta_sinal(sinal)}'
                f'<div class="banco-preco">{preco} <span style="margin-left:6px;">{variacao_html}</span></div>'
                f'{confianca_html}</div>')

    @staticmethod
    def cartao_cenario(titulo: str, valor: str, variacao_html: str, tipo: str, detalhe: str = "") -> str:
        """Cenário de preço (pessimista, esperado ou otimista)."""
        detalhe_html = f'<div class="cartao-detalhe">{detalhe}</div>' if detalhe else ""
        return (f'<div class="cenario cenario-{tipo}"><div class="cartao-rotulo">{titulo}</div>'
                f'<div class="cartao-valor">{valor}</div><div>{variacao_html}</div>{detalhe_html}</div>')

    @staticmethod
    def banner(sinal: str | None, titulo: str, destaque: str, info_html: str) -> str:
        classe = sinal if sinal in ROTULOS_SINAL else "neutro"
        return (f'<div class="banner banner-{classe}"><div class="banner-titulo">{titulo}</div>'
                f'<div class="banner-sinal">{destaque}</div><div class="banner-info">{info_html}</div></div>')

    # ── Listas ─────────────────────────────────────────────────────────────
    @staticmethod
    def linha(titulo_html: str, detalhe_html: str, direita_html: str) -> str:
        return (f'<div class="linha"><div><div class="linha-titulo">{titulo_html}</div>'
                f'<div class="linha-detalhe">{detalhe_html}</div></div>'
                f'<div class="linha-direita">{direita_html}</div></div>')

    @classmethod
    def noticia(cls, titulo: str, link: str, fonte: str, publicada_em: datetime | None, ticker: str) -> str:
        quando = f"{publicada_em:%d/%m às %H:%M}" if publicada_em else ""
        meta = " · ".join(parte for parte in (escape(fonte), quando) if parte)
        link_seguro = escape(link, quote=True) if link.startswith(("http://", "https://")) else "#"
        return (f'<div class="noticia"><div class="noticia-titulo">'
                f'<a href="{link_seguro}" target="_blank" rel="noopener noreferrer">{escape(titulo)}</a></div>'
                f'<div class="noticia-meta">{cls.etiqueta(ticker, "info")} {meta}</div></div>')

    @staticmethod
    def grade(cartoes: list[str], colunas: int = 4) -> str:
        """
        Cartões lado a lado com a mesma altura (grade CSS).

        As colunas do Streamlit não igualam a altura do conteúdo; a grade sim,
        o que deixa as fileiras de cartões alinhadas.
        """
        return f'<div class="grade grade-{colunas}">{"".join(cartoes)}</div>'

    @staticmethod
    def cartao_html(conteudo_html: str, rotulo: str = "") -> str:
        """Cartão genérico com título opcional."""
        rotulo_html = f'<div class="rotulo-secao">{rotulo}</div>' if rotulo else ""
        return f'<div class="cartao">{rotulo_html}{conteudo_html}</div>'
