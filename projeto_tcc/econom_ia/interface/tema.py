"""
Identidade visual: paleta de cores, logotipo e folha de estilos (tema escuro).

As cores das séries dos gráficos seguem uma paleta categórica validada para
daltonismo sobre o fundo escuro; compra e venda usam cores de estado (verde e
vermelho) sempre acompanhadas de texto ou ícone, nunca apenas pela cor.
"""

from __future__ import annotations

import streamlit as st


class Paleta:
    """Cores do tema escuro."""

    FUNDO = "#0b0f14"
    FUNDO_LATERAL = "#0f1318"
    CARTAO = "#161a23"
    BORDA = "#1f2430"
    BORDA_FORTE = "#2a2f3a"
    TEXTO = "#f1f5f9"
    TEXTO_SECUNDARIO = "#94a3b8"
    TEXTO_FRACO = "#64748b"

    # Estados dos sinais (sempre com rótulo textual).
    COMPRA = "#0ca30c"
    VENDA = "#d03b3b"
    NEUTRO = "#94a3b8"
    ALERTA = "#fab219"

    # Paleta categórica (ordem fixa) para ativos e modelos.
    SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]
    DESTAQUE = "#3987e5"
    PREVISAO = "#c98500"

    @classmethod
    def cor_sinal(cls, sinal: str | None) -> str:
        return {"compra": cls.COMPRA, "venda": cls.VENDA}.get(sinal or "", cls.NEUTRO)


LOGO_SVG = """
<svg width="34" height="34" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Logotipo Econom-IA">
  <defs>
    <linearGradient id="gradEconomia" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#3987e5"/>
      <stop offset="1" stop-color="#199e70"/>
    </linearGradient>
  </defs>
  <rect x="1" y="1" width="32" height="32" rx="8" fill="url(#gradEconomia)"/>
  <polyline points="7,23 13,17 18,20 26,10" fill="none" stroke="#ffffff" stroke-width="2.4"
            stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="13" cy="17" r="2.2" fill="#ffffff"/>
  <circle cx="18" cy="20" r="2.2" fill="#ffffff"/>
  <circle cx="26" cy="10" r="2.6" fill="#ffffff"/>
</svg>
"""

# Logotipo com o nome, exibido no alto da barra lateral (st.logo).
LOGO_MARCA_SVG = (
    LOGO_SVG.strip().replace('width="34" height="34" viewBox="0 0 34 34"',
                             'width="176" height="34" viewBox="0 0 176 34"')
    .replace("</svg>", '<text x="44" y="24" font-family="Source Sans Pro, Segoe UI, Arial, sans-serif" '
                       'font-size="21" font-weight="800" fill="#f1f5f9">Econom-IA</text></svg>')
)

CSS = f"""
<style>
/* ── Base: letra maior e mais espaço para leitura ─────────────────────── */
html, body, [class*="st-"] {{ font-size: 16px; }}
.stApp {{ background-color: {Paleta.FUNDO}; }}
.block-container {{ padding-top: 2.6rem !important; padding-bottom: 1rem !important; max-width: 1320px; }}
p, li, [data-testid="stMarkdownContainer"] {{ font-size: 16px; line-height: 1.6; }}
[data-testid="stCaptionContainer"], .stCaption {{ font-size: 14.5px !important; line-height: 1.55;
                                                  color: {Paleta.TEXTO_SECUNDARIO} !important; }}
label, [data-testid="stWidgetLabel"] p {{ font-size: 15px !important; color: {Paleta.TEXTO_SECUNDARIO} !important; }}

/* ── Menu na barra lateral (st.navigation) ────────────────────────────── */
/* O botão de recolher (« ) e o de reabrir (» ) ficam sempre visíveis: sem eles,
   uma barra recolhida deixaria o usuário sem caminho para as outras telas. */
#MainMenu, footer, [data-testid="stMainMenu"] {{ display: none; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background-color: {Paleta.FUNDO_LATERAL}; border-right: 1px solid {Paleta.BORDA}; }}
[data-testid="stSidebarNav"] a span {{ font-size: 16px; }}
[data-testid="stSidebarNav"] header, [data-testid="stNavSectionHeader"] {{ font-size: 13px; letter-spacing: .3px; }}
[data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] {{ visibility: visible !important;
                                                                                  display: flex !important; }}
[data-testid="stLogo"] {{ height: 2.4rem !important; max-width: 13rem; }}
/* Com a barra recolhida, o botão de reabrir ganha destaque e o rótulo "Menu" */
[data-testid="stExpandSidebarButton"] {{ background: #1d4f8c; border: 1px solid {Paleta.DESTAQUE};
                                         border-radius: 8px; padding: 2px 12px 2px 8px; }}
[data-testid="stExpandSidebarButton"]::after {{ content: "Menu"; font-size: 15px; font-weight: 600;
                                                color: {Paleta.TEXTO}; margin-left: 4px; }}
.secao-lateral {{ font-size: 13px; color: {Paleta.TEXTO_SECUNDARIO}; font-weight: 700; margin: 22px 0 8px 0; }}

/* ── Cabeçalho de cada tela ───────────────────────────────────────────── */
.titulo-pagina {{ font-size: 30px; font-weight: 700; color: {Paleta.TEXTO}; line-height: 1.2; margin-bottom: 6px; }}
.subtitulo-pagina {{ font-size: 16.5px; color: {Paleta.TEXTO_SECUNDARIO}; margin-bottom: 22px; max-width: 760px;
                     line-height: 1.5; }}
.titulo-secao {{ font-size: 21px; font-weight: 700; color: {Paleta.TEXTO}; margin: 34px 0 4px 0; line-height: 1.3; }}
.titulo-secao-detalhe {{ display: block; font-size: 15px; font-weight: 400; color: {Paleta.TEXTO_SECUNDARIO};
                         margin: 2px 0 12px 0; }}
.titulo-secao:not(:has(.titulo-secao-detalhe)) {{ margin-bottom: 14px; }}
.rotulo-secao {{ font-size: 17px; color: {Paleta.TEXTO}; font-weight: 700; margin-bottom: 10px; }}

/* ── Grades de cartões com a mesma altura ─────────────────────────────── */
.grade {{ display: grid; gap: 16px; margin-bottom: 16px; }}
.grade > div {{ margin-bottom: 0 !important; }}
.grade-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.grade-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.grade-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
@media (max-width: 1000px) {{ .grade-3, .grade-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 620px) {{ .grade-2, .grade-3, .grade-4 {{ grid-template-columns: minmax(0, 1fr); }}
                             .titulo-pagina {{ font-size: 25px; }} }}

/* ── Cartões ──────────────────────────────────────────────────────────── */
.cartao {{ background: {Paleta.CARTAO}; border: 1px solid {Paleta.BORDA}; border-radius: 12px;
           padding: 20px 22px; margin-bottom: 16px; }}
.cartao-rotulo {{ font-size: 15px; color: {Paleta.TEXTO_SECUNDARIO}; font-weight: 600; }}
.cartao-valor {{ font-size: 30px; font-weight: 700; color: {Paleta.TEXTO}; margin: 6px 0 4px 0; line-height: 1.15; }}
.cartao-valor-texto {{ font-size: 22px; line-height: 1.3; }}
.cartao-unidade {{ font-size: 16px; color: {Paleta.TEXTO_SECUNDARIO}; font-weight: 400; }}
.cartao-detalhe {{ font-size: 15px; color: #cbd5e1; line-height: 1.45; }}
.cartao-fonte {{ font-size: 13.5px; color: {Paleta.TEXTO_FRACO}; margin-top: 10px; }}
.sobe {{ color: #34d399; font-weight: 600; }}
.desce {{ color: #f87171; font-weight: 600; }}
.estavel {{ color: {Paleta.TEXTO_SECUNDARIO}; }}

/* ── Abertura da Visão Geral ──────────────────────────────────────────── */
.boas-vindas {{ background: {Paleta.CARTAO}; border: 1px solid {Paleta.BORDA}; border-radius: 14px;
                padding: 26px 28px; margin-bottom: 20px; }}
.boas-vindas-data {{ font-size: 15px; color: {Paleta.TEXTO_SECUNDARIO}; margin-bottom: 8px; }}
.boas-vindas-titulo {{ font-size: 32px; font-weight: 700; color: {Paleta.TEXTO}; line-height: 1.25; }}
.boas-vindas-detalhe {{ font-size: 16.5px; color: #cbd5e1; margin-top: 10px; line-height: 1.5; }}

/* ── Resposta principal de cada tela (banner) ─────────────────────────── */
.banner {{ border-radius: 14px; padding: 24px 28px; margin-bottom: 20px; border: 1px solid {Paleta.BORDA};
           border-left-width: 6px; }}
.banner-compra {{ background: #122417; border-color: {Paleta.COMPRA}; }}
.banner-venda {{ background: #2a1616; border-color: {Paleta.VENDA}; }}
.banner-neutro {{ background: #1a1f29; border-color: #475569; }}
.banner-titulo {{ font-size: 16px; font-weight: 600; color: {Paleta.TEXTO_SECUNDARIO}; }}
.banner-sinal {{ font-size: 34px; font-weight: 800; margin: 8px 0 10px 0; color: {Paleta.TEXTO}; line-height: 1.2; }}
.banner-info {{ font-size: 16.5px; line-height: 1.7; color: #dbe3ee; max-width: 980px; }}
.banner-confianca {{ font-size: 17px; font-weight: 600; color: {Paleta.TEXTO_SECUNDARIO}; margin-left: 12px; }}

/* ── Etiquetas de sinal (sempre com texto) ────────────────────────────── */
.etiqueta {{ display: inline-block; padding: 4px 11px; border-radius: 7px; font-weight: 700; font-size: 14px;
             white-space: nowrap; line-height: 1.35; }}
.etiqueta-compra {{ background: rgba(12,163,12,.18); color: #34d399; }}
.etiqueta-venda {{ background: rgba(208,59,59,.2); color: #f87171; }}
.etiqueta-neutro {{ background: rgba(148,163,184,.16); color: #cbd5e1; }}
.etiqueta-info {{ background: rgba(57,135,229,.18); color: #8fc0f5; }}
.etiqueta-aviso {{ background: rgba(250,178,25,.15); color: {Paleta.ALERTA}; }}

/* ── Linhas de listas (sinais de hoje, notícias) ──────────────────────── */
.linha {{ display: flex; justify-content: space-between; align-items: center; gap: 14px;
          padding: 14px 0; border-bottom: 1px solid {Paleta.BORDA}; }}
.linha:last-child {{ border-bottom: none; }}
.linha-titulo {{ font-size: 17px; font-weight: 700; color: {Paleta.TEXTO}; }}
.linha-detalhe {{ font-size: 15px; color: {Paleta.TEXTO_SECUNDARIO}; font-weight: 400; }}
.noticia-titulo a {{ color: {Paleta.TEXTO}; text-decoration: none; font-size: 16.5px; font-weight: 600;
                     line-height: 1.45; }}
.noticia-titulo a:hover, .noticia-titulo a:focus-visible {{ color: #8fc0f5; text-decoration: underline; }}
.noticia-meta {{ font-size: 14px; color: {Paleta.TEXTO_FRACO}; margin-top: 6px; }}

/* ── Sinal de cada banco (Visão Geral) ────────────────────────────────── */
.banco {{ background: {Paleta.CARTAO}; border: 1px solid {Paleta.BORDA}; border-top-width: 4px;
          border-radius: 12px; padding: 18px 20px; }}
.banco-compra {{ border-top-color: {Paleta.COMPRA}; }}
.banco-venda {{ border-top-color: {Paleta.VENDA}; }}
.banco-neutro {{ border-top-color: #475569; }}
.banco-ticker {{ font-size: 20px; font-weight: 800; color: {Paleta.TEXTO}; }}
.banco-nome {{ font-size: 15px; color: {Paleta.TEXTO_SECUNDARIO}; margin-bottom: 12px; }}
.banco-preco {{ font-size: 17px; color: {Paleta.TEXTO}; margin: 12px 0 2px 0; }}
.banco-confianca {{ font-size: 14px; color: {Paleta.TEXTO_FRACO}; margin-top: 4px; }}

/* ── Indicadores técnicos ─────────────────────────────────────────────── */
.cartao-indicador {{ min-height: 150px; }}
.consenso-marcas {{ min-height: 26px; margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; }}
.consenso-sinal {{ margin: 10px 0; }}
.indicador-topo {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;
                   margin-bottom: 6px; flex-wrap: wrap; }}
.indicador-nome {{ font-size: 16px; color: {Paleta.TEXTO}; font-weight: 700; }}
.indicador-valor {{ font-size: 26px; font-weight: 700; color: {Paleta.TEXTO}; margin: 8px 0; }}
.indicador-desc {{ font-size: 14.5px; color: {Paleta.TEXTO_SECUNDARIO}; line-height: 1.5; }}

/* ── Cenários de preço ────────────────────────────────────────────────── */
.cenario {{ background: {Paleta.CARTAO}; border: 1px solid {Paleta.BORDA}; border-left-width: 5px;
            border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; }}
.cenario-pessimista {{ border-left-color: {Paleta.VENDA}; }}
.cenario-esperado {{ border-left-color: {Paleta.PREVISAO}; }}
.cenario-otimista {{ border-left-color: {Paleta.COMPRA}; }}

/* ── Manchetes e rodapé de sinais ─────────────────────────────────────── */
.noticia {{ padding: 14px 0; border-bottom: 1px solid {Paleta.BORDA}; }}
.noticia:first-child {{ padding-top: 0; }}
.noticia:last-child {{ border-bottom: none; padding-bottom: 0; }}
.linha-direita {{ display: flex; align-items: center; gap: 10px; text-align: right; }}
.rodape-sinais {{ display: flex; justify-content: space-around; padding-top: 12px; margin-top: 8px;
                  border-top: 1px solid {Paleta.BORDA}; font-size: 15px; }}

/* ── Avisos e notas ───────────────────────────────────────────────────── */
.aviso-titulo {{ color: {Paleta.ALERTA}; }}
.aviso {{ background: rgba(250,178,25,.07); border-left: 4px solid {Paleta.ALERTA}; padding: 14px 18px;
          border-radius: 8px; font-size: 15px; color: #cbd5e1; margin-top: 28px; line-height: 1.55; }}
.nota {{ background: rgba(57,135,229,.08); border-left: 4px solid {Paleta.DESTAQUE}; padding: 14px 18px;
         border-radius: 8px; font-size: 15.5px; color: #dbe3ee; margin: 8px 0 16px 0; line-height: 1.6; }}
.texto-corrido {{ font-size: 16px; line-height: 1.7; color: #dbe3ee; max-width: 980px; }}
.lista-resumo {{ margin: 0; padding-left: 20px; }}
.lista-resumo li {{ margin-bottom: 8px; }}
.lista-resumo li:last-child {{ margin-bottom: 0; }}

/* ── Situação do sistema e créditos (barra lateral) ───────────────────── */
.status {{ border: 1px solid {Paleta.BORDA}; background: {Paleta.CARTAO}; padding: 12px 14px; border-radius: 10px;
           margin-bottom: 10px; }}
.status-titulo {{ font-size: 14px; font-weight: 700; }}
.status-texto {{ font-size: 13.5px; color: #cbd5e1; margin-top: 4px; line-height: 1.5; }}
.creditos {{ margin-top: 16px; padding-top: 12px; border-top: 1px solid {Paleta.BORDA}; font-size: 13px;
             color: {Paleta.TEXTO_FRACO}; line-height: 1.6; }}
.creditos strong {{ color: {Paleta.TEXTO_SECUNDARIO}; }}
.aviso-lateral {{ display: block; margin-top: 6px; color: {Paleta.ALERTA}; }}

/* ── Componentes nativos ──────────────────────────────────────────────── */
[data-testid="stMetric"] {{ background: {Paleta.CARTAO}; border: 1px solid {Paleta.BORDA}; border-radius: 12px;
                           padding: 16px 20px; }}
[data-testid="stMetricLabel"] p {{ color: {Paleta.TEXTO_SECUNDARIO} !important; font-size: 15px !important; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{ font-size: 28px !important; }}
[data-testid="stExpander"] details {{ border-color: {Paleta.BORDA}; border-radius: 12px; background: {Paleta.CARTAO}; }}
[data-testid="stExpander"] summary p {{ font-size: 16px; font-weight: 600; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 16px; }}
.stButton > button {{ background: {Paleta.BORDA}; color: {Paleta.TEXTO}; border: 1px solid {Paleta.BORDA_FORTE};
                      border-radius: 8px; min-height: 2.75rem; }}
.stButton > button p {{ font-size: 15.5px; }}
.stButton > button:hover {{ background: {Paleta.BORDA_FORTE}; border-color: {Paleta.TEXTO_FRACO}; }}
.stButton > button[kind="primary"] {{ background: #1d4f8c; border-color: {Paleta.DESTAQUE}; }}
:focus-visible {{ outline: 2px solid {Paleta.DESTAQUE} !important; outline-offset: 2px; }}
</style>
"""


def aplicar_tema() -> None:
    """Injeta a folha de estilos na página."""
    st.markdown(CSS, unsafe_allow_html=True)
