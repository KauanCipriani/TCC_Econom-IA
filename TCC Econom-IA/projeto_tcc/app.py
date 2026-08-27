"""
==============================================================================
TCC - Sistema de Apoio à Decisão para Predição de Tendências de Ativos da B3
==============================================================================
Autores: Kauan Amélio Cipriani e Vitor Hugo Konzen
Curso: Ciência da Computação - UNOESC São Miguel do Oeste
Orientador: Vinicius Almeida Santos
==============================================================================

FASE 5 — INTERFACE WEB INTERATIVA (Streamlit)

Aplicação web que apresenta as análises dos modelos de Machine Learning em
cinco telas integradas:

    1. Visão Geral      — panorama do setor financeiro e das quatro ações.
    2. Análise Detalhada — gráficos, sinais e previsão LSTM por ativo.
    3. Performance      — métricas dos modelos e validação out-of-sample.
    4. Centro de Alertas — histórico tabular de sinais com filtros.
    5. Sobre o Projeto  — informações sobre os autores, orientador e pipeline.

A aplicação opera em dois modos automaticamente:

    • MODO REAL  — quando os modelos treinados (Fases 3 e 4) estão disponíveis
                   na pasta modelos/. Usa os modelos reais para gerar sinais e
                   predições.
    • MODO DEMO  — quando os modelos ainda não foram treinados. Gera sinais
                   por regras heurísticas baseadas em RSI e MACD. Permite
                   demonstrar a interface antes da finalização do treinamento.

Como executar:
    streamlit run app.py

A aplicação abrirá automaticamente em http://localhost:8501
==============================================================================
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────────────────────────────────────
TICKERS_INFO = {
    "ITUB4":  {"nome": "Itaú Unibanco",      "color": "#1e40af"},
    "BBDC4":  {"nome": "Bradesco",           "color": "#dc2626"},
    "BBAS3":  {"nome": "Banco do Brasil",    "color": "#16a34a"},
    "SANB11": {"nome": "Santander Brasil",   "color": "#f59e0b"},
}

# Períodos selecionáveis nos gráficos (em pregões aprox.; None = todo o histórico)
PERIODOS_GRAFICO = {
    "Último mês": 22,
    "Últimos 6 meses": 126,
    "Último ano": 252,
    "Tudo (desde 2019)": None,
}

# Features na MESMA ordem usada no treinamento do Random Forest (treinar_modelos.py)
FEATURES_MODELO = [
    "rsi_14", "macd_hist_z",
    "sma_9_ratio", "sma_21_ratio", "sma_50_ratio",
    "ema_9_ratio", "ema_21_ratio",
    "bb_position",
]

# Caminhos relativos ao proprio arquivo, para funcionar tanto localmente
# quanto em hospedagem (ex.: Streamlit Community Cloud), onde o diretorio de
# trabalho e a raiz do repositorio.
BASE_DIR = Path(__file__).resolve().parent
DADOS_PATH = BASE_DIR / "dados"
MODELOS_PATH = BASE_DIR / "modelos"
ARQUIVO_FEATURES = DADOS_PATH / "b3_financeiro_features.parquet"
ARQUIVO_OHLCV = DADOS_PATH / "b3_financeiro_ohlcv.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# CSS CUSTOMIZADO — replica o visual profissional dos mockups
# ─────────────────────────────────────────────────────────────────────────────
def aplicar_css() -> None:
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Barra lateral fixa */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] { display: none !important; }
    [data-testid="stSidebar"] {
        background-color: #0f1318; border-right: 1px solid #1f2430;
        transform: none !important; visibility: visible !important;
        min-width: 244px !important; width: 244px !important;
    }
    /* Garante o conteudo da sidebar sempre visivel */
    [data-testid="stSidebar"][aria-expanded="false"] { margin-left: 0 !important; }
    [data-testid="stSidebar"] > div { visibility: visible !important; }

    .stApp { background-color: #0b0f14; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1200px; }

    .page-title { font-size: 22px; font-weight: 700; color: #f1f5f9; margin-bottom: 2px; }
    .page-subtitle { font-size: 13px; color: #94a3b8; margin-bottom: 18px; }

    /* Cards planos */
    .custom-card, .indice-card, .indicator-card, .news-card, .economia-card, .hero-welcome {
        background: #161a23; border: 1px solid #1f2430; border-radius: 8px;
        padding: 16px; margin-bottom: 12px;
    }

    /* Hero de boas-vindas (plano) */
    .hero-welcome { padding: 20px; }
    .hero-date { font-size: 13px; color: #94a3b8; margin-bottom: 6px; }
    .hero-title { font-size: 22px; font-weight: 700; color: #f1f5f9; line-height: 1.3; }
    .hero-accent-up { color: #16a34a; }
    .hero-accent-down { color: #dc2626; }
    .hero-accent-neutral { color: #94a3b8; }

    /* Indices macro */
    .indice-label { font-size: 11px; color: #64748b; font-weight: 600; }
    .indice-value { font-size: 20px; font-weight: 700; color: #f1f5f9; margin: 4px 0; }
    .indice-unit { font-size: 12px; color: #94a3b8; font-weight: 400; }
    .indice-delta-up { font-size: 12px; color: #16a34a; font-weight: 600; }
    .indice-delta-down { font-size: 12px; color: #dc2626; font-weight: 600; }
    .indice-delta-stable { font-size: 12px; color: #94a3b8; }

    /* Banner de sinal (plano, sem gradiente) */
    .alert-banner-buy, .alert-banner-sell, .alert-banner-neutral {
        border-radius: 8px; padding: 18px; margin-bottom: 14px; border: 1px solid #1f2430;
    }
    .alert-banner-buy { background: #14241b; border-color: #16a34a; color: #d1fae5; }
    .alert-banner-sell { background: #2a1718; border-color: #dc2626; color: #fecaca; }
    .alert-banner-neutral { background: #1a1f29; border-color: #4b5563; color: #cbd5e1; }
    .alert-banner-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
    .alert-banner-signal { font-size: 28px; font-weight: 700; margin: 6px 0; }
    .alert-banner-info { font-size: 13px; line-height: 1.6; }

    /* Pills */
    .pill-buy { background: rgba(22,163,74,.15); color: #16a34a; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }
    .pill-sell { background: rgba(220,38,38,.15); color: #dc2626; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }
    .pill-neutral { background: rgba(148,163,184,.15); color: #94a3b8; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }

    /* Indicadores tecnicos */
    .indicator-name { font-size: 11px; color: #94a3b8; font-weight: 600; text-transform: uppercase; }
    .indicator-value { font-size: 20px; font-weight: 700; color: #f1f5f9; margin: 6px 0; }
    .indicator-desc { font-size: 11px; color: #94a3b8; }

    /* Noticias (placeholder) */
    .news-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; margin-right: 6px; background: #1f2430; color: #94a3b8; }
    .news-title { font-size: 14px; font-weight: 600; color: #f1f5f9; margin: 8px 0 6px 0; }
    .news-resumo { font-size: 12px; color: #94a3b8; line-height: 1.5; }
    .news-meta { font-size: 11px; color: #64748b; margin-top: 8px; }

    /* Linhas de sinais */
    .signal-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1f2430; }
    .signal-row:last-child { border-bottom:none; }
    .signal-ticker { font-size:13px; font-weight:600; color:#f1f5f9; }
    .signal-price { font-size:12px; color:#94a3b8; margin-left:8px; }

    /* Card / badge de desenvolvimento */
    .economia-badge { display:inline-block; background:#1f2430; color:#94a3b8; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:700; letter-spacing:1px; }
    .economia-title { font-size:16px; font-weight:700; color:#f1f5f9; margin:8px 0; }
    .economia-desc { font-size:12px; color:#cbd5e1; line-height:1.5; }
    .economia-button { background:#1f2430; color:#f1f5f9; padding:10px; border-radius:6px; text-align:center; font-weight:600; font-size:13px; margin-top:12px; }

    .alert-warning { background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.25); color:#cbd5e1; padding:12px; border-radius:6px; font-size:12px; margin-top:12px; }

    .sidebar-section-title { font-size:10px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin:14px 0 6px 0; }

    div[data-testid="stRadio"] > label { display: none; }

    [data-testid="stMetric"] { background:#161a23; border:1px solid #1f2430; border-radius:8px; padding:12px; }
    [data-testid="stMetricLabel"] { color:#94a3b8 !important; }
    [data-testid="stMetricValue"] { color:#f1f5f9 !important; }

    .stButton > button { background:#1f2430; color:#f1f5f9; border:1px solid #2a2f3a; border-radius:6px; }
    .stButton > button:hover { background:#2a2f3a; border-color:#64748b; }

    [data-testid="stDataFrame"] { background:#161a23; border-radius:8px; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS
# ─────────────────────────────────────────────────────────────────────────────
# Mapeia os tickers da B3 para os símbolos do Yahoo Finance (sufixo .SA)
TICKERS_YF = {
    "ITUB4":  "ITUB4.SA",
    "BBDC4":  "BBDC4.SA",
    "BBAS3":  "BBAS3.SA",
    "SANB11": "SANB11.SA",
}


@st.cache_data(ttl=900, show_spinner="Atualizando cotações da B3...")
def baixar_dados_online() -> Optional[pd.DataFrame]:
    """
    Baixa o histórico diário ATUALIZADO das quatro ações direto do Yahoo
    Finance (via yfinance) e calcula os indicadores técnicos. O resultado
    fica em cache por 15 minutos (ttl=900), de modo que as cotações são
    renovadas constantemente sem precisar reiniciar o aplicativo.

    Retorna None se não houver conexão ou dados disponíveis.
    """
    try:
        import yfinance as yf
    except Exception:
        return None

    frames = []
    for ticker, simbolo in TICKERS_YF.items():
        try:
            # Histórico diário desde jan/2019 (conforme o TCC) até a data atual
            hist = yf.download(
                simbolo, start="2019-01-01", interval="1d",
                auto_adjust=True, progress=False, threads=False,
            )
        except Exception:
            continue
        if hist is None or hist.empty:
            continue
        # Com um único símbolo o yfinance pode devolver colunas MultiIndex
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist.reset_index().rename(columns={
            "Date": "data", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        hist["ticker"] = ticker
        colunas = ["data", "open", "high", "low", "close", "volume", "ticker"]
        frames.append(hist[colunas])

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"])
    df = df.dropna(subset=["close"])
    df = df.sort_values(["ticker", "data"]).reset_index(drop=True)
    return adicionar_indicadores_basicos(df)


@st.cache_data(ttl=900, show_spinner=False)
def carregar_dados() -> tuple[Optional[pd.DataFrame], str]:
    """
    Carrega os dados das ações. Primeiro tenta baixar cotações atualizadas
    do Yahoo Finance (modo 'online'); se não houver conexão, recorre aos
    arquivos Parquet locais ('features' ou 'ohlcv') como cópia de segurança.
    Retorna (DataFrame, modo).
    """
    df_online = baixar_dados_online()
    if df_online is not None and not df_online.empty:
        return df_online, "online"

    if ARQUIVO_FEATURES.exists():
        df = pd.read_parquet(ARQUIVO_FEATURES)
        df["data"] = pd.to_datetime(df["data"])
        return df.sort_values(["ticker", "data"]).reset_index(drop=True), "features"

    if ARQUIVO_OHLCV.exists():
        df = pd.read_parquet(ARQUIVO_OHLCV)
        df["data"] = pd.to_datetime(df["data"])
        df = df.sort_values(["ticker", "data"]).reset_index(drop=True)
        df = adicionar_indicadores_basicos(df)
        return df, "ohlcv"

    return None, "erro"


def adicionar_indicadores_basicos(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula RSI, MACD e SMAs simples para o modo demo (sem pandas-ta)."""
    saida = []
    for ticker, grupo in df.groupby("ticker"):
        g = grupo.copy().reset_index(drop=True)

        # RSI(14)
        delta = g["close"].diff()
        ganho = delta.where(delta > 0, 0.0).rolling(14).mean()
        perda = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = ganho / perda.replace(0, np.nan)
        g["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = g["close"].ewm(span=12, adjust=False).mean()
        ema26 = g["close"].ewm(span=26, adjust=False).mean()
        g["macd_line"] = ema12 - ema26
        g["macd_signal"] = g["macd_line"].ewm(span=9, adjust=False).mean()
        g["macd_hist"] = g["macd_line"] - g["macd_signal"]

        # Médias móveis simples (SMA) e exponenciais (EMA) de 9, 21 e 50 períodos
        for periodo in (9, 21, 50):
            g[f"sma_{periodo}"] = g["close"].rolling(periodo).mean()
            g[f"ema_{periodo}"] = g["close"].ewm(span=periodo, adjust=False).mean()

        # Bandas de Bollinger
        sma20 = g["close"].rolling(20).mean()
        std20 = g["close"].rolling(20).std()
        g["bb_upper"] = sma20 + 2 * std20
        g["bb_middle"] = sma20
        g["bb_lower"] = sma20 - 2 * std20

        # Features do modelo (razões e z-score), espelhando treinar_modelos.py
        for periodo in (9, 21, 50):
            g[f"sma_{periodo}_ratio"] = g["close"] / g[f"sma_{periodo}"] - 1
        for periodo in (9, 21):
            g[f"ema_{periodo}_ratio"] = g["close"] / g[f"ema_{periodo}"] - 1
        g["bb_position"] = (g["close"] - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"])
        g["macd_hist_z"] = (
            (g["macd_hist"] - g["macd_hist"].rolling(60).mean())
            / g["macd_hist"].rolling(60).std()
        )

        saida.append(g)
    return pd.concat(saida, ignore_index=True)


def _formatar_br(valor: float, casas: int = 2) -> str:
    """Formata um número no padrão brasileiro (1.234,56)."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(ttl=900, show_spinner=False)
def carregar_indicadores_macro() -> list:
    """
    Busca, ao vivo, os indicadores macroeconômicos exibidos na Visão Geral:
      • IBOVESPA e USD/BRL  → Yahoo Finance (yfinance)
      • SELIC (meta) e IPCA → API pública do Banco Central do Brasil (SGS)
    O resultado fica em cache por 15 min (ttl=900), atualizando constantemente.
    Retorna uma lista de tuplas (label, valor, unidade, delta, direção).
    """
    import json
    import urllib.request

    def yf_ultimo(simbolo: str):
        try:
            import yfinance as yf
            h = yf.download(simbolo, period="5d", interval="1d",
                            auto_adjust=True, progress=False, threads=False)
            if h is None or h.empty:
                return None
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = h.columns.get_level_values(0)
            fech = h["Close"].dropna()
            if len(fech) < 2:
                return None
            atual, anterior = float(fech.iloc[-1]), float(fech.iloc[-2])
            return atual, (atual / anterior - 1) * 100
        except Exception:
            return None

    def bcb_serie(serie: int):
        try:
            url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}"
                   f"/dados/ultimos/1?formato=json")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                dados = json.load(resp)
            return float(dados[-1]["valor"])
        except Exception:
            return None

    indices = []

    # IBOVESPA (^BVSP)
    ibov = yf_ultimo("^BVSP")
    if ibov:
        valor, var = ibov
        direcao = "up" if var >= 0 else "down"
        sinal = "+" if var >= 0 else ""
        indices.append(("IBOVESPA", _formatar_br(valor, 0), "pts",
                        f"{sinal}{_formatar_br(var, 2)}%", direcao))
    else:
        indices.append(("IBOVESPA", "—", "pts", "indisponível", "stable"))

    # SELIC meta (BCB série 432)
    selic = bcb_serie(432)
    if selic is not None:
        indices.append(("SELIC", _formatar_br(selic, 2), "% a.a.",
                        "meta Copom", "stable"))
    else:
        indices.append(("SELIC", "—", "% a.a.", "indisponível", "stable"))

    # USD/BRL (BRL=X)
    usd = yf_ultimo("BRL=X")
    if usd:
        valor, var = usd
        direcao = "up" if var >= 0 else "down"
        sinal = "+" if var >= 0 else ""
        indices.append(("USD/BRL", "R$ " + _formatar_br(valor, 2), "",
                        f"{sinal}{_formatar_br(var, 2)}%", direcao))
    else:
        indices.append(("USD/BRL", "—", "", "indisponível", "stable"))

    # IPCA acumulado em 12 meses (BCB série 13522)
    ipca = bcb_serie(13522)
    if ipca is not None:
        indices.append(("IPCA", _formatar_br(ipca, 2), "% (12m)",
                        "acum. 12 meses", "stable"))
    else:
        indices.append(("IPCA", "—", "% (12m)", "indisponível", "stable"))

    return indices


@st.cache_resource(show_spinner=False)
def carregar_modelos() -> dict:
    """Tenta carregar os modelos treinados. Se falhar, retorna dict vazio."""
    modelos = {}
    try:
        import joblib  # noqa: F401
        if (MODELOS_PATH / "random_forest.pkl").exists():
            modelos["rf"] = joblib.load(MODELOS_PATH / "random_forest.pkl")
        if (MODELOS_PATH / "svm.pkl").exists():
            modelos["svm"] = joblib.load(MODELOS_PATH / "svm.pkl")
            modelos["scaler_svm"] = joblib.load(MODELOS_PATH / "scaler_svm.pkl")
    except Exception:
        pass
    return modelos


@st.cache_data(show_spinner=False)
def carregar_metricas() -> Optional[dict]:
    """Carrega o arquivo de métricas reais geradas pelo treinamento."""
    import json
    arquivo = MODELOS_PATH / "metricas.json"
    if not arquivo.exists():
        return None
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GERADOR DE SINAIS — MODO DEMO (heurístico baseado em indicadores)
# ─────────────────────────────────────────────────────────────────────────────
def gerar_sinal_heuristico(linha: pd.Series) -> tuple[str, float]:
    """
    Gera sinal de compra/venda/neutro com base em regras técnicas simples.
    Retorna (sinal, confiança 0-100).

    Regras:
        - RSI < 35 + MACD_hist > 0      → forte compra
        - RSI < 45 + MACD positivo      → compra leve
        - RSI > 65 + MACD_hist < 0      → forte venda
        - RSI > 55 + MACD negativo      → venda leve
        - Caso contrário                → neutro
    """
    rsi = linha.get("rsi_14", 50)
    macd_hist = linha.get("macd_hist", 0)

    if pd.isna(rsi) or pd.isna(macd_hist):
        return "neutro", 50.0

    if rsi < 35 and macd_hist > 0:
        return "compra", 85.0
    if rsi < 45 and macd_hist > 0:
        return "compra", 70.0
    if rsi > 65 and macd_hist < 0:
        return "venda", 82.0
    if rsi > 55 and macd_hist < 0:
        return "venda", 68.0
    return "neutro", 55.0


def adicionar_sinais_modelo(df: pd.DataFrame, modelos: dict) -> pd.DataFrame:
    """
    Gera os sinais de compra/venda/neutro com o modelo Random Forest treinado,
    adicionando as colunas 'sinal' e 'confianca' (0-100) ao DataFrame. As linhas
    sem todas as features (warm-up dos indicadores) ficam sem sinal e recaem na
    heurística via obter_sinal().
    """
    df = df.copy()
    df["sinal"] = pd.NA
    df["confianca"] = pd.NA

    rf = modelos.get("rf")
    if rf is None or not all(c in df.columns for c in FEATURES_MODELO):
        return df

    mask = df[FEATURES_MODELO].notna().all(axis=1)
    if mask.any():
        X = df.loc[mask, FEATURES_MODELO].values
        df.loc[mask, "sinal"] = rf.predict(X)
        df.loc[mask, "confianca"] = (rf.predict_proba(X).max(axis=1) * 100).round(0)
    return df


def obter_sinal(linha: pd.Series) -> tuple[str, float]:
    """
    Retorna (sinal, confiança). Usa a previsão do Random Forest (colunas 'sinal'
    e 'confianca' geradas por adicionar_sinais_modelo). Se não houver previsão
    para a linha (warm-up ou modelo ausente), recorre à heurística RSI/MACD.
    """
    sinal = linha.get("sinal")
    conf = linha.get("confianca")
    if isinstance(sinal, str) and conf is not None and not pd.isna(conf):
        return sinal, float(conf)
    return gerar_sinal_heuristico(linha)


def previsao_lstm_demo(df_ticker: pd.DataFrame, dias: int) -> pd.DataFrame:
    """
    Previsão demo: extrapola tendência recente com pequena suavização e
    inclui faixa de confiança que cresce com o horizonte.
    """
    ultimos = df_ticker.tail(30)
    tendencia = (ultimos["close"].iloc[-1] - ultimos["close"].iloc[0]) / len(ultimos)
    preco_atual = ultimos["close"].iloc[-1]

    datas_futuras = pd.bdate_range(
        start=df_ticker["data"].iloc[-1] + timedelta(days=1), periods=dias
    )

    precos = []
    for i in range(1, dias + 1):
        # Suaviza a tendência ao longo do tempo (reversão à média)
        fator_suavizacao = 0.5
        precos.append(preco_atual + tendencia * i * fator_suavizacao)

    precos = np.array(precos)
    incerteza = 0.015 * np.arange(1, dias + 1)
    return pd.DataFrame({
        "data": datas_futuras,
        "previsao": precos,
        "lim_inf": precos * (1 - incerteza),
        "lim_sup": precos * (1 + incerteza),
    })


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS PLOTLY
# ─────────────────────────────────────────────────────────────────────────────
def grafico_principal(df_ticker: pd.DataFrame, df_previsao: pd.DataFrame, ticker: str,
                      dias: Optional[int] = None) -> go.Figure:
    """Gráfico principal: histórico + médias móveis + previsão LSTM."""
    df_recente = df_ticker if dias is None else df_ticker.tail(dias)
    fig = go.Figure()

    # Preço histórico
    fig.add_trace(go.Scatter(
        x=df_recente["data"], y=df_recente["close"],
        mode="lines", name="Preço de fechamento",
        line=dict(color="#1e40af", width=2.5),
    ))

    # SMA 21
    if "sma_21" in df_recente.columns:
        fig.add_trace(go.Scatter(
            x=df_recente["data"], y=df_recente["sma_21"],
            mode="lines", name="SMA 21",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
        ))

    # Sinais de compra e venda (heurísticos)
    sinais = df_recente.apply(lambda r: obter_sinal(r)[0], axis=1)
    mask_compra = sinais == "compra"
    mask_venda = sinais == "venda"

    fig.add_trace(go.Scatter(
        x=df_recente.loc[mask_compra, "data"],
        y=df_recente.loc[mask_compra, "close"],
        mode="markers", name="Sinal de compra",
        marker=dict(symbol="triangle-up", size=12, color="#16a34a"),
    ))
    fig.add_trace(go.Scatter(
        x=df_recente.loc[mask_venda, "data"],
        y=df_recente.loc[mask_venda, "close"],
        mode="markers", name="Sinal de venda",
        marker=dict(symbol="triangle-down", size=12, color="#dc2626"),
    ))

    # Previsão LSTM
    if not df_previsao.empty:
        # Faixa de confiança
        fig.add_trace(go.Scatter(
            x=list(df_previsao["data"]) + list(df_previsao["data"][::-1]),
            y=list(df_previsao["lim_sup"]) + list(df_previsao["lim_inf"][::-1]),
            fill="toself", fillcolor="rgba(245,158,11,0.18)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            name="Faixa de confiança",
        ))
        fig.add_trace(go.Scatter(
            x=df_previsao["data"], y=df_previsao["previsao"],
            mode="lines+markers", name="Previsão LSTM",
            line=dict(color="#f59e0b", width=2.5, dash="dash"),
            marker=dict(size=7),
        ))

    fig.update_layout(
        title=dict(text=f"Histórico, sinais e previsão — {ticker}", font=dict(size=15, color="#f1f5f9")),
        xaxis_title="Data", yaxis_title="Preço (R$)",
        hovermode="x unified", height=480,
        paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
        font=dict(color="#cbd5e1"),
        xaxis=dict(gridcolor="#1f2430", color="#94a3b8"),
        yaxis=dict(gridcolor="#1f2430", color="#94a3b8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def grafico_setor(df: pd.DataFrame) -> go.Figure:
    """Gráfico consolidado: variação % de cada ação nos últimos 180 dias."""
    fig = go.Figure()
    for ticker, info in TICKERS_INFO.items():
        sub = df[df["ticker"] == ticker].tail(180)
        if sub.empty:
            continue
        base = sub["close"].iloc[0]
        variacao = (sub["close"] / base - 1) * 100
        fig.add_trace(go.Scatter(
            x=sub["data"], y=variacao, mode="lines", name=ticker,
            line=dict(color=info["color"], width=2.5),
        ))
    fig.update_layout(
        title=dict(text="Variação % consolidada do setor financeiro", font=dict(size=14, color="#f1f5f9")),
        xaxis_title="Data", yaxis_title="Variação (%)",
        hovermode="x unified", height=350,
        paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
        font=dict(color="#cbd5e1"),
        xaxis=dict(gridcolor="#1f2430", color="#94a3b8"),
        yaxis=dict(gridcolor="#1f2430", color="#94a3b8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTES VISUAIS
# ─────────────────────────────────────────────────────────────────────────────
def banner_alerta(sinal: str, ticker: str, preco_atual: float, preco_previsto: float, confianca: float) -> None:
    """Banner colorido grande no topo da análise detalhada."""
    variacao_pct = (preco_previsto - preco_atual) / preco_atual * 100

    if sinal == "compra":
        classe = "alert-banner-buy"
        emoji = ""
        titulo = "SINAL DE COMPRA DETECTADO"
    elif sinal == "venda":
        classe = "alert-banner-sell"
        emoji = ""
        titulo = "SINAL DE VENDA DETECTADO"
    else:
        classe = "alert-banner-neutral"
        emoji = ""
        titulo = "TENDÊNCIA NEUTRA"

    st.markdown(f"""
    <div class="{classe}">
        <div class="alert-banner-title">{emoji} {titulo} — {ticker}</div>
        <div class="alert-banner-signal">{sinal.upper()}</div>
        <div class="alert-banner-info">
            <strong>Preço atual:</strong> R$ {preco_atual:.2f} &nbsp;•&nbsp;
            <strong>Previsão:</strong> R$ {preco_previsto:.2f} ({variacao_pct:+.2f}%) &nbsp;•&nbsp;
            <strong>Confiança:</strong> {confianca:.0f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


def card_indicador(nome: str, valor: str, descricao: str, status: str = "neutro") -> str:
    """Renderiza um card de indicador técnico (RSI, MACD, etc.)."""
    cores_status = {
        "alta":   ("#dcfce7", "#15803d", "ALTA"),
        "baixa":  ("#fee2e2", "#dc2626", "BAIXA"),
        "neutro": ("#f1f5f9", "#475569", "NEUTRO"),
    }
    bg, fg, label = cores_status[status]
    return f"""
    <div class="indicator-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span class="indicator-name">{nome}</span>
            <span style="background:{bg}; color:{fg}; padding:2px 8px; border-radius:8px; font-size:10px; font-weight:700;">{label}</span>
        </div>
        <div class="indicator-value">{valor}</div>
        <div class="indicator-desc">{descricao}</div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# TELAS
# ─────────────────────────────────────────────────────────────────────────────
def tela_inicio(df: pd.DataFrame) -> None:
    """Nova tela de boas-vindas — primeiro contato do usuário com o sistema."""
    from datetime import datetime

    agora = datetime.now()
    hora = agora.hour
    if hora < 12:
        saudacao = "Bom dia"
    elif hora < 18:
        saudacao = "Boa tarde"
    else:
        saudacao = "Boa noite"

    # Determina humor do mercado pela variação média do setor
    variacoes_dia = []
    for ticker in TICKERS_INFO:
        sub = df[df["ticker"] == ticker]
        if len(sub) >= 2:
            v = (sub.iloc[-1]["close"] - sub.iloc[-2]["close"]) / sub.iloc[-2]["close"] * 100
            variacoes_dia.append(v)
    var_media = sum(variacoes_dia) / len(variacoes_dia) if variacoes_dia else 0

    if var_media > 0.5:
        humor = "em alta."
        humor_class = "hero-accent-up"
    elif var_media < -0.5:
        humor = "em baixa."
        humor_class = "hero-accent-down"
    else:
        humor = "estável."
        humor_class = "hero-accent-neutral"

    dia_semana = {
        "Monday": "Segunda-feira", "Tuesday": "Terça-feira",
        "Wednesday": "Quarta-feira", "Thursday": "Quinta-feira",
        "Friday": "Sexta-feira", "Saturday": "Sábado",
        "Sunday": "Domingo",
    }[agora.strftime("%A")]
    meses = ["", "janeiro", "fevereiro", "março", "abril", "maio", "junho",
             "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
    data_formatada = f"{dia_semana}, {agora.day} de {meses[agora.month]} de {agora.year}"

    # ── HERO BANNER DE BOAS-VINDAS ────────────────────────────────────
    st.markdown(f"""
    <div class="hero-welcome">
        <div class="hero-date">{data_formatada}</div>
        <div class="hero-title">{saudacao}! O mercado está <span class="{humor_class}">{humor}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 CARDS DE ÍNDICES MACROECONÔMICOS (ao vivo) ──────────────────
    col1, col2, col3, col4 = st.columns(4)

    indices = carregar_indicadores_macro()

    for col, (label, valor, unit, delta, direcao) in zip([col1, col2, col3, col4], indices):
        delta_class = {"up": "indice-delta-up", "down": "indice-delta-down",
                       "stable": "indice-delta-stable"}[direcao]
        arrow = {"up": "▲", "down": "▼", "stable": "●"}[direcao]
        with col:
            st.markdown(f"""
            <div class="indice-card">
                <div class="indice-label">{label}</div>
                <div class="indice-value">{valor} <span class="indice-unit">{unit}</span></div>
                <div class="{delta_class}">{arrow} {delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.caption("Fontes ao vivo: Yahoo Finance (IBOVESPA, USD/BRL) e Banco Central do Brasil (SELIC, IPCA) · atualiza a cada 15 min")
    st.markdown("---")

    # ── LAYOUT EM 2 COLUNAS: NOTÍCIAS + SIDEBAR DIREITA ───────────────
    col_noticias, col_lateral = st.columns([2.3, 1])

    with col_noticias:
        st.markdown("### Últimas Notícias")
        st.markdown("""
        <div style="background:#161a23;
                    border:1px solid #1f2430; border-radius:12px; padding:32px; text-align:center;">
            <span class="economia-badge">EM DESENVOLVIMENTO · TC II</span>
            <h3 style="color:#f1f5f9; margin-top:12px; font-size:20px;">Feed de notícias em construção</h3>
            <p style="color:#cbd5e1; line-height:1.6; margin-top:8px; font-size:13px;">
                A integração com APIs de notícias do mercado financeiro está prevista
                para a entrega final do TCC (TC II).
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── SIDEBAR DIREITA: Acesso Rápido + Sinais de Hoje + EconomIA ────
    with col_lateral:
        # Acesso Rápido
        st.markdown("""
        <div class="custom-card">
            <div style="font-size:11px; color:#64748b; font-weight:700; letter-spacing:1px;">ACESSO RÁPIDO</div>
            <div style="margin-top:12px;">
                <div style="background:#161a23; border:1px solid #1f2430; border-radius:8px; padding:12px; margin-bottom:8px;">
                    <div style="font-size:13px; font-weight:700; color:#10b981;">Painel Iniciante</div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Simples e visual</div>
                </div>
                <div style="background:#161a23; border:1px solid #1f2430; border-radius:8px; padding:12px; margin-bottom:8px;">
                    <div style="font-size:13px; font-weight:700; color:#60a5fa;">Painel Avançado</div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Técnico e preditivo</div>
                </div>
                <div style="background:#161a23; border:1px solid #1f2430; border-radius:8px; padding:12px;">
                    <div style="font-size:13px; font-weight:700; color:#c084fc;">EconomIA</div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Pergunte à IA</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Sinais de Hoje
        sinais_partes = ['<div style="font-size:11px; color:#64748b; font-weight:700; letter-spacing:1px; margin-bottom:12px;">SINAIS DE HOJE</div>']
        contadores = {"compra": 0, "venda": 0, "neutro": 0}
        for ticker, info in TICKERS_INFO.items():
            sub = df[df["ticker"] == ticker]
            if sub.empty:
                continue
            ultimo = sub.iloc[-1]
            anterior = sub.iloc[-2] if len(sub) > 1 else ultimo
            var = (ultimo["close"] - anterior["close"]) / anterior["close"] * 100
            sinal, _ = obter_sinal(ultimo)
            contadores[sinal] += 1

            pill_class = f"pill-{'buy' if sinal == 'compra' else 'sell' if sinal == 'venda' else 'neutral'}"
            var_class = "indice-delta-up" if var >= 0 else "indice-delta-down"
            var_str = f"+{var:.2f}%" if var >= 0 else f"{var:.2f}%"
            row = (
                f'<div class="signal-row">'
                f'<div><span class="signal-ticker">{ticker}</span>'
                f'<span class="signal-price">R$ {ultimo["close"]:.2f}</span></div>'
                f'<div style="display:flex; align-items:center; gap:8px;">'
                f'<span class="{var_class}" style="font-size:11px; font-weight:700;">{var_str}</span>'
                f'<span class="{pill_class}">{sinal.upper()}</span>'
                f'</div></div>'
            )
            sinais_partes.append(row)

        rodape = (
            f'<div style="display:flex; justify-content:space-around; padding-top:12px; margin-top:8px; border-top:1px solid #1f2430; font-size:11px;">'
            f'<div style="color:#10b981; font-weight:700;">{contadores["compra"]} COMPRA</div>'
            f'<div style="color:#f59e0b; font-weight:700;">{contadores["neutro"]} NEUTRO</div>'
            f'<div style="color:#ef4444; font-weight:700;">{contadores["venda"]} VENDA</div>'
            f'</div>'
        )
        sinais_partes.append(rodape)
        sinais_html = "".join(sinais_partes)

        st.markdown(f'<div class="custom-card">{sinais_html}</div>', unsafe_allow_html=True)

        # Card EconomIA promocional
        st.markdown("""
        <div class="economia-card">
            <span class="economia-badge">IA · TC II</span>
            <div class="economia-title">EconomIA</div>
            <div class="economia-desc">
                Pergunte sobre qualquer ação do setor bancário, indicadores técnicos ou conjuntura econômica.
            </div>
            <div class="economia-button">Iniciar conversa →</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Aviso de plataforma educacional ───────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(245,158,11,0.08); border-left:3px solid #f59e0b;
                padding:12px 16px; border-radius:6px; margin-top:12px;">
        <strong style="color:#fbbf24;">Aviso importante:</strong>
        <span style="color:#cbd5e1; font-size:13px;">
            Esta plataforma é educacional. Não constitui recomendação de investimento. Consulte um assessor credenciado pela CVM antes de tomar decisões financeiras.
        </span>
    </div>
    """, unsafe_allow_html=True)


def tela_economia() -> None:
    """Placeholder da IA conversacional — entrega prevista para o TC II."""
    st.markdown('<div class="page-title">EconomIA — Assistente Conversacional</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Funcionalidade prevista para o TC II — entrega final do TCC</div>',
        unsafe_allow_html=True,
    )

    # Banner explicativo
    st.markdown("""
    <div style="background:#161a23;
                border:1px solid #1f2430; border-radius:12px; padding:32px; margin-bottom:24px;">
        <span class="economia-badge">EM DESENVOLVIMENTO · TC II</span>
        <h2 style="color:#f1f5f9; margin-top:12px; font-size:24px;">
            Pergunte à IA sobre o mercado
        </h2>
        <p style="color:#cbd5e1; line-height:1.7; margin-top:8px; font-size:14px;">
            A EconomIA será um assistente conversacional integrado ao modelo Claude (Anthropic),
            capaz de responder perguntas em linguagem natural sobre os quatro ativos monitorados
            pelo sistema, explicar indicadores técnicos e contextualizar os sinais gerados pelos
            algoritmos de Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Prévia: exemplos de perguntas que serão suportadas
    st.markdown("### Exemplos de perguntas que a EconomIA responderá")

    col_a, col_b = st.columns(2)

    exemplos_iniciante = [
        ("Pergunta", "Por que o sistema sugeriu compra para ITUB4 hoje?"),
        ("EconomIA", "O sistema identificou três indicadores apontando para uma possível tendência de alta no ITUB4. O MACD cruzou a linha de sinal para cima há dois dias, padrão historicamente associado a movimentos altistas. O preço atual está no terço superior das Bandas de Bollinger, indicando força do movimento. E o RSI em 58 mostra que ainda há espaço para subida. Lembre-se: isto é apoio à decisão, não recomendação."),
    ]

    exemplos_expert = [
        ("Pergunta", "Qual feature mais pesou no sinal de compra de hoje?"),
        ("EconomIA", "Na execução de hoje, a feature de maior contribuição foi o macd_hist_z (z-score do histograma do MACD), com 16,61% de importância no Random Forest. As três features de maior peso foram: macd_hist_z (16,61%), sma_9_ratio (12,76%) e rsi_14 (12,56%). Em conjunto representam mais de 41% das decisões do modelo nas últimas 24 horas."),
    ]

    with col_a:
        st.markdown("""
        <div class="custom-card">
            <div style="font-size:11px; color:#64748b; font-weight:700; letter-spacing:1px;">EXEMPLO · MODO INICIANTE</div>
        """, unsafe_allow_html=True)
        for autor, texto in exemplos_iniciante:
            cor = "#94a3b8" if "Pergunta" in autor else "#c084fc"
            st.markdown(f"""
            <div style="margin-top:12px; padding:12px; background:#0f1318; border-radius:8px;">
                <div style="font-size:11px; color:{cor}; font-weight:700; margin-bottom:6px;">{autor}</div>
                <div style="color:#cbd5e1; font-size:13px; line-height:1.5;">{texto}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="custom-card">
            <div style="font-size:11px; color:#64748b; font-weight:700; letter-spacing:1px;">EXEMPLO · MODO AVANÇADO</div>
        """, unsafe_allow_html=True)
        for autor, texto in exemplos_expert:
            cor = "#94a3b8" if "Pergunta" in autor else "#c084fc"
            st.markdown(f"""
            <div style="margin-top:12px; padding:12px; background:#0f1318; border-radius:8px;">
                <div style="font-size:11px; color:{cor}; font-weight:700; margin-bottom:6px;">{autor}</div>
                <div style="color:#cbd5e1; font-size:13px; line-height:1.5;">{texto}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Arquitetura técnica
    st.markdown("### Como funcionará")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div style="font-size:24px;">1</div>
            <div style="font-weight:700; color:#f1f5f9; margin-top:6px;">Usuário faz a pergunta</div>
            <div style="color:#94a3b8; font-size:12px; margin-top:4px;">Em linguagem natural, sobre qualquer um dos 4 ativos monitorados.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="custom-card">
            <div style="font-size:24px;">2</div>
            <div style="font-weight:700; color:#f1f5f9; margin-top:6px;">Sistema busca contexto</div>
            <div style="color:#94a3b8; font-size:12px; margin-top:4px;">Recupera dados atuais dos Parquet/JSON: preço, indicadores, sinal gerado.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="custom-card">
            <div style="font-size:24px;">3</div>
            <div style="font-weight:700; color:#f1f5f9; margin-top:6px;">Claude responde</div>
            <div style="color:#94a3b8; font-size:12px; margin-top:4px;">Combina o contexto recuperado com seu conhecimento de finanças (técnica RAG).</div>
        </div>
        """, unsafe_allow_html=True)

    # Salvaguardas
    st.markdown("### Salvaguardas éticas e regulatórias")
    st.markdown("""
    <div class="custom-card">
        <ul style="color:#cbd5e1; line-height:1.8; margin:0;">
            <li>Limitado <strong>exclusivamente</strong> aos quatro ativos monitorados (ITUB4, BBDC4, BBAS3, SANB11)</li>
            <li>Recusará perguntas sobre outros ativos fora do escopo do sistema</li>
            <li><strong>Nunca</strong> emitirá recomendações diretas de compra ou venda</li>
            <li>Toda resposta incluirá lembrete de que o sistema é apenas apoio à decisão</li>
            <li>Cumpre as diretrizes da CVM sobre não-recomendação por agentes não credenciados</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Status
    st.info(
        "**Status atual:** funcionalidade documentada na seção 3.9 do TCC, "
        "com referências teóricas a Brown et al. (2020), Lewis et al. (2020) e Vaswani et al. (2017). "
        "Implementação técnica prevista para a entrega final (TC II)."
    )


def tela_previsoes_macro() -> None:
    """Previsoes & Macro - entrega prevista para o TC II."""
    st.markdown('<div class="page-title">Previsoes e Macroeconomia</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Indicadores macro e sentimento de mercado - entrega prevista para o TC II</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Funcionalidade em desenvolvimento. A integracao automatica de indicadores "
        "macroeconomicos (Selic, IPCA, Fed Funds, Treasury Yields) e do mercado de "
        "previsao Polymarket esta prevista para a entrega final do TCC (TC II). "
        "Estes dados sao apenas contextuais e nao influenciam os sinais de compra e "
        "venda, que sao gerados pelo modelo Random Forest a partir dos indicadores tecnicos."
    )


def tela_economia_funcional(df: pd.DataFrame, modelos: dict, metricas: Optional[dict]) -> None:
    """EconomIA funcional — responde perguntas básicas com dados reais do sistema."""
    st.markdown('<div class="page-title">EconomIA — Assistente</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Demo • Responde perguntas sobre os 4 ativos monitorados • '
        'Versão completa com Claude API prevista para o TC II</div>',
        unsafe_allow_html=True,
    )

    # Aviso sobre o estado atual
    st.warning(
        "**Versão demonstrativa em desenvolvimento.** Esta primeira versão responde a um "
        "conjunto limitado de perguntas usando dados reais do sistema. A integração completa "
        "com Claude API (Anthropic) está prevista para o TC II, permitindo perguntas em "
        "linguagem natural sobre qualquer aspecto dos ativos monitorados."
    )

    # Estado de chat
    if "chat_historico" not in st.session_state:
        st.session_state.chat_historico = []

    # Sugestões de perguntas
    st.markdown("### Perguntas que a EconomIA já responde")
    st.caption("Clique em qualquer uma para obter a resposta:")

    perguntas_suportadas = [
        "Qual ação está com sinal de compra hoje?",
        "Como o sistema gera os sinais?",
        "Qual ação subiu mais no último mês?",
        "O que é o indicador RSI?",
        "Qual a precisão atual do modelo?",
        "Por que SANB11 é usada para validação?",
        "Devemos comprar ITUB4?",
        "O que é o Índice de Basileia?",
    ]

    # Botões em grid 2x4
    cols_q = st.columns(4)
    for i, pergunta in enumerate(perguntas_suportadas):
        with cols_q[i % 4]:
            if st.button(pergunta, key=f"q_{i}", use_container_width=True):
                resposta = responder_pergunta(pergunta, df, modelos, metricas)
                st.session_state.chat_historico.append({"pergunta": pergunta, "resposta": resposta})

    # Campo de pergunta livre
    st.markdown("---")
    st.markdown("### Faça sua pergunta")
    pergunta_livre = st.text_input(
        "Digite sua pergunta:",
        placeholder="Ex: Como funciona o MACD?",
        key="pergunta_livre",
    )
    if st.button("Perguntar à EconomIA", type="primary"):
        if pergunta_livre.strip():
            resposta = responder_pergunta(pergunta_livre, df, modelos, metricas)
            st.session_state.chat_historico.append({"pergunta": pergunta_livre, "resposta": resposta})

    # Exibe histórico
    if st.session_state.chat_historico:
        st.markdown("---")
        st.markdown("### Conversa")
        # Inverte para mostrar mais recente em cima
        for item in reversed(st.session_state.chat_historico[-10:]):
            st.markdown(f"""
            <div style="background:#1f2430; border-left:3px solid #94a3b8; padding:12px 16px; border-radius:8px; margin-bottom:8px;">
                <div style="font-size:11px; color:#94a3b8; font-weight:700; margin-bottom:4px;">VOCÊ</div>
                <div style="color:#f1f5f9; font-size:13px;">{item['pergunta']}</div>
            </div>
            <div style="background:#161a23; border-left:3px solid #1f2430; padding:14px 16px; border-radius:8px; margin-bottom:16px;">
                <div style="font-size:11px; color:#c084fc; font-weight:700; margin-bottom:6px;">ECONOMIA</div>
                <div style="color:#e5e7eb; font-size:13px; line-height:1.6;">{item['resposta']}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Limpar conversa"):
            st.session_state.chat_historico = []
            st.rerun()


def responder_pergunta(pergunta: str, df: pd.DataFrame, modelos: dict, metricas: Optional[dict]) -> str:
    """Sistema de respostas baseado em palavras-chave + dados reais do sistema."""
    p = pergunta.lower()

    # Identifica ações mencionadas
    tickers_mencionados = [t for t in TICKERS_INFO if t.lower() in p]

    # ── PERGUNTAS SOBRE SINAIS ──────────────────────────────────────
    if any(w in p for w in ["sinal de compra", "compra hoje", "compras hoje", "comprar hoje"]):
        compras = []
        for ticker in TICKERS_INFO:
            sub = df[df["ticker"] == ticker]
            if sub.empty:
                continue
            sinal, conf = obter_sinal(sub.iloc[-1])
            if sinal == "compra":
                compras.append(f"<b>{ticker}</b> ({conf:.0f}% confiança)")

        if compras:
            return ("As ações com sinal de <b>COMPRA</b> hoje são: " + ", ".join(compras) +
                   ". Esses sinais foram gerados com base em uma combinação de RSI baixo "
                   "e MACD positivo. Lembre-se: o sistema é apoio à decisão, não recomendação.")
        return ("No momento, <b>nenhuma das ações monitoradas</b> apresenta sinal claro de "
                "compra. Os indicadores estão majoritariamente em zona neutra. "
                "Aguardar movimentos mais definidos pode ser prudente.")

    if any(w in p for w in ["sinal de venda", "venda hoje", "vendas hoje", "vender"]):
        vendas = []
        for ticker in TICKERS_INFO:
            sub = df[df["ticker"] == ticker]
            if sub.empty:
                continue
            sinal, conf = obter_sinal(sub.iloc[-1])
            if sinal == "venda":
                vendas.append(f"<b>{ticker}</b> ({conf:.0f}% confiança)")
        if vendas:
            return ("Ações com sinal de <b>VENDA</b>: " + ", ".join(vendas) +
                   ". Esses sinais combinam RSI alto (sobrecompra) e MACD negativo.")
        return "No momento não há sinais claros de venda para nenhuma das ações monitoradas."

    # ── PERGUNTAS SOBRE COMO O SISTEMA FUNCIONA ─────────────────────
    if any(w in p for w in ["como o sistema gera", "como funciona o sistema", "como gera os sinais",
                            "como gera sinal", "como a ia"]):
        return ("O sistema usa um modelo de <b>Machine Learning</b> chamado <b>Random Forest</b>, "
                "treinado com 7 anos de dados históricos das ações ITUB4, BBDC4 e BBAS3 (a SANB11 "
                "foi reservada para validação). Para cada pregão, ele calcula <b>indicadores "
                "técnicos</b> como RSI, MACD, Bandas de Bollinger e Médias Móveis. O modelo aprendeu "
                "a identificar combinações desses indicadores que historicamente precederam "
                "movimentos de alta (compra) ou baixa (venda). Quando esses padrões aparecem nos "
                "dados de hoje, o sistema gera o sinal correspondente.")

    # ── PERGUNTAS SOBRE DESEMPENHO/MÊS ──────────────────────────────
    if "último mês" in p or "ultimo mes" in p or "subiu mais" in p or "caiu mais" in p:
        variacoes = []
        for ticker in TICKERS_INFO:
            sub = df[df["ticker"] == ticker]
            if len(sub) < 30:
                continue
            var = (sub.iloc[-1]["close"] - sub.iloc[-30]["close"]) / sub.iloc[-30]["close"] * 100
            variacoes.append((ticker, var))
        if variacoes:
            variacoes.sort(key=lambda x: x[1], reverse=True)
            melhor, pior = variacoes[0], variacoes[-1]
            return (f"No último mês, a ação com melhor desempenho foi a <b>{melhor[0]}</b> "
                   f"com variação de <b>{melhor[1]:+.2f}%</b>. A pior foi <b>{pior[0]}</b> "
                   f"com <b>{pior[1]:+.2f}%</b>. Lembre-se que retorno passado não garante "
                   "rentabilidade futura.")

    # ── PERGUNTAS EDUCACIONAIS ──────────────────────────────────────
    if "rsi" in p:
        return ("O <b>RSI (Índice de Força Relativa)</b> é um indicador técnico que varia de 0 a 100 "
                "e mede a velocidade e a magnitude das mudanças de preço. Valores acima de <b>70</b> "
                "indicam que a ação pode estar <b>sobrecomprada</b> (possível correção à vista). "
                "Valores abaixo de <b>30</b> indicam <b>sobrevenda</b> (possível alta à vista). "
                "Entre 30 e 70 é zona neutra. Foi criado por J. Welles Wilder em 1978.")

    if "macd" in p:
        return ("O <b>MACD (Moving Average Convergence Divergence)</b> compara duas médias móveis "
                "exponenciais do preço (12 e 26 períodos). Quando a linha MACD cruza a linha de "
                "sinal <b>para cima</b>, sugere início de tendência de alta. Quando cruza <b>para "
                "baixo</b>, sugere tendência de baixa. É um dos indicadores mais importantes do "
                "nosso modelo (16,6% de contribuição nas decisões do Random Forest).")

    if "bollinger" in p or "bandas" in p:
        return ("As <b>Bandas de Bollinger</b> são três linhas: uma média móvel central (geralmente "
                "de 20 períodos) e duas bandas — superior e inferior — que ficam a 2 desvios-padrão "
                "da média. Quando o preço toca a banda superior, pode estar sobrecomprado; quando "
                "toca a inferior, sobrevendido. As bandas se contraem em períodos de baixa "
                "volatilidade e se expandem em alta volatilidade.")

    if "basileia" in p or "basiléia" in p:
        return ("O <b>Índice de Basileia</b> mede a solidez de um banco, comparando seu Patrimônio "
                "de Referência (PR) com seus Ativos Ponderados pelo Risco (RWA). No Brasil, o "
                "Banco Central exige mínimo de <b>10,5%</b>. Os quatro bancos monitorados ficam "
                "geralmente acima de <b>15%</b>, considerados muito sólidos. Quanto maior o índice, "
                "mais capital o banco tem para absorver perdas. A integração completa desses dados "
                "fundamentalistas via API Brapi está prevista para o TC II.")

    if "roe" in p:
        return ("O <b>ROE (Retorno sobre o Patrimônio Líquido)</b> mede quanto lucro o banco gera "
                "para cada real de patrimônio dos acionistas. Em bancos privados brasileiros como "
                "Itaú e Bradesco, o ROE costuma ser superior a <b>18% ao ano</b>. O Banco do "
                "Brasil tende a ter ROE mais baixo por ter mais exposição ao crédito agro e "
                "menor flexibilidade operacional (banco público). Esses dados serão integrados "
                "via API no TC II.")

    # ── PERGUNTAS SOBRE PERFORMANCE/MÉTRICAS ────────────────────────
    if any(w in p for w in ["precisão", "precisao", "f1", "acurácia", "acuracia", "métrica"]):
        if metricas:
            f1_oos = metricas["resultados_oos"]["f1_macro"]
            acc_oos = metricas["resultados_oos"]["accuracy"]
            return (f"O modelo Random Forest atingiu, no último treinamento "
                   f"({metricas['info_treinamento']['data_treinamento'][:10]}): "
                   f"<b>F1-Score de {f1_oos:.4f}</b> e <b>acurácia de {acc_oos:.4f}</b> "
                   "no conjunto out-of-sample (SANB11, nunca visto pelo modelo). "
                   "<br><br><b>Importante:</b> esse F1 ainda está próximo de uma escolha "
                   "aleatória (0,33 para 3 classes). É o resultado esperado nesta fase do "
                   "TC I — estão previstos ajustes substanciais para o TC II: GridSearchCV, "
                   "novas features (volume, volatilidade), comparação com SVM e LSTM.")
        return ("As métricas reais do modelo serão exibidas após o treinamento. Execute "
                "<code>python treinar_modelos.py</code> para gerar.")

    # ── PERGUNTAS SOBRE VALIDAÇÃO OUT-OF-SAMPLE ─────────────────────
    if "sanb11" in p and ("validação" in p or "validacao" in p or "teste" in p or "por que" in p or "porque" in p):
        return ("A <b>SANB11 (Santander)</b> foi reservada inteiramente como conjunto de "
                "<b>validação out-of-sample</b>. Isso significa que ela nunca foi vista pelo modelo "
                "durante o treinamento — apenas ITUB4, BBDC4 e BBAS3 foram usadas para treinar. "
                "Essa estratégia testa se o modelo realmente <b>aprendeu padrões dos indicadores</b> "
                "(generalizáveis) ou se apenas <b>memorizou movimentos específicos</b> dos ativos "
                "treinados. Se o desempenho no SANB11 for parecido com o do treino, o modelo "
                "generalizou bem.")

    # ── PERGUNTAS DIRETAS SOBRE AÇÕES ESPECÍFICAS ───────────────────
    if tickers_mencionados and ("devo comprar" in p or "devemos comprar" in p or
                                 "comprar" in p or "vender" in p):
        ticker = tickers_mencionados[0]
        sub = df[df["ticker"] == ticker]
        if not sub.empty:
            sinal, conf = obter_sinal(sub.iloc[-1])
            preco = sub.iloc[-1]["close"]
            return (f"<b>Importante:</b> o sistema não emite recomendações de compra ou venda. "
                   f"Posso te dizer apenas o que os indicadores estão mostrando agora: "
                   f"<br><br>Para <b>{ticker}</b> ({TICKERS_INFO[ticker]['nome']}), o sinal "
                   f"atual do modelo é <b>{sinal.upper()}</b> com {conf:.0f}% de confiança, "
                   f"com o preço em R$ {preco:.2f}. "
                   f"<br><br>Esta análise é baseada apenas em indicadores técnicos e <b>não "
                   "constitui recomendação de investimento</b>. Para decisões de compra ou "
                   "venda, consulte um assessor credenciado pela CVM.")

    if tickers_mencionados:
        ticker = tickers_mencionados[0]
        sub = df[df["ticker"] == ticker]
        if not sub.empty:
            sinal, conf = obter_sinal(sub.iloc[-1])
            preco = sub.iloc[-1]["close"]
            return (f"Sobre <b>{ticker}</b> ({TICKERS_INFO[ticker]['nome']}): preço atual "
                   f"R$ {preco:.2f}, sinal do modelo <b>{sinal.upper()}</b> "
                   f"({conf:.0f}% confiança). Para análise detalhada, vá no <b>Painel "
                   "Avançado</b> e selecione essa ação.")

    # ── FALLBACK ──────────────────────────────────────────────────
    return ("Ainda não fui treinada para responder essa pergunta específica. "
            "A versão completa (TC II), integrada ao Claude (Anthropic), conseguirá "
            "responder perguntas em linguagem natural sobre qualquer aspecto do sistema. "
            "<br><br>Por enquanto, experimente uma das <b>perguntas sugeridas acima</b> ou "
            "reformule sua pergunta usando palavras-chave como: RSI, MACD, Bollinger, "
            "Basileia, ROE, sinais de compra, performance, SANB11, ou o nome de uma ação "
            "(ITUB4, BBDC4, BBAS3, SANB11).")


def tela_visao_geral(df: pd.DataFrame, modo: str) -> None:
    """Tela de visão geral simplificada (referência legada)."""
    st.markdown('<div class="page-title">Visão Geral do Setor Financeiro — B3</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Panorama atual das quatro principais ações de bancos brasileiros</div>',
        unsafe_allow_html=True,
    )

    # Estatísticas gerais do setor
    col1, col2, col3, col4 = st.columns(4)
    sinais_resumo = {"compra": 0, "venda": 0, "neutro": 0}
    for ticker in TICKERS_INFO:
        sub = df[df["ticker"] == ticker].tail(1)
        if not sub.empty:
            sinal, _ = obter_sinal(sub.iloc[0])
            sinais_resumo[sinal] += 1

    col1.metric("Sugestões de compra hoje", sinais_resumo["compra"])
    col2.metric("Sugestões de cautela hoje", sinais_resumo["venda"])
    col3.metric("Sem sinal claro", sinais_resumo["neutro"])
    col4.metric("Ações acompanhadas", len(TICKERS_INFO))

    st.markdown("---")
    st.markdown("### Ações que o sistema acompanha")
    colunas = st.columns(4)
    for i, (ticker, info) in enumerate(TICKERS_INFO.items()):
        sub = df[df["ticker"] == ticker].copy()
        if sub.empty:
            continue
        ultimo = sub.iloc[-1]
        anterior = sub.iloc[-2] if len(sub) > 1 else ultimo
        variacao = (ultimo["close"] - anterior["close"]) / anterior["close"] * 100
        sinal, conf = obter_sinal(ultimo)
        cor_sinal = "" if sinal == "compra" else "" if sinal == "venda" else ""
        descricao = {"compra": "Momento favorável",
                     "venda": "Sinal de cautela",
                     "neutro": "Sem direção clara"}[sinal]

        with colunas[i]:
            with st.container(border=True):
                st.markdown(f"**{ticker}** &nbsp; {cor_sinal}")
                st.caption(info["nome"])
                st.metric(label="Preço", value=f"R$ {ultimo['close']:.2f}", delta=f"{variacao:+.2f}%")
                st.caption(f"Avaliação: **{descricao}**")

    st.markdown("---")
    st.plotly_chart(grafico_setor(df), use_container_width=True)


def tela_analise_iniciante(df: pd.DataFrame, modelos: dict) -> None:
    """Versão simplificada da análise — linguagem leiga, sem jargão técnico."""

    # Seleção do ativo — só dropdown, sem slider de previsão (default 7 dias)
    ticker = st.selectbox(
        "Qual ação você quer analisar?",
        options=list(TICKERS_INFO.keys()),
        format_func=lambda t: f"{TICKERS_INFO[t]['nome']}  ({t})",
        key="ticker_iniciante",
    )

    df_ticker = df[df["ticker"] == ticker].copy().reset_index(drop=True)
    if df_ticker.empty:
        st.warning(f"Sem dados para {ticker}.")
        return

    ultimo = df_ticker.iloc[-1]
    sinal, confianca = obter_sinal(ultimo)
    df_previsao = previsao_lstm_demo(df_ticker, 7)
    preco_atual = float(ultimo["close"])
    preco_previsto = float(df_previsao["previsao"].iloc[-1]) if not df_previsao.empty else preco_atual

    # Variação esperada para os próximos 7 pregões
    var_esperada_pct = (preco_previsto - preco_atual) / preco_atual * 100

    # ── BANNER PRINCIPAL — pergunta + resposta clara ─────────────────────
    nome_acao = TICKERS_INFO[ticker]["nome"]

    if sinal == "compra":
        resposta_curta = "Pode ser um bom momento"
        cor_classe = "alert-banner-buy"
        emoji = ""
        explicacao = (
            f"Os sinais técnicos sugerem que <strong>{nome_acao}</strong> pode estar em um "
            "momento favorável. O modelo identificou uma combinação de fatores "
            "que historicamente indicam possibilidade de valorização."
        )
    elif sinal == "venda":
        resposta_curta = "Talvez seja melhor ter cautela"
        cor_classe = "alert-banner-sell"
        emoji = ""
        explicacao = (
            f"Os sinais técnicos sugerem que <strong>{nome_acao}</strong> pode estar em um "
            "momento de risco. O modelo identificou indicadores que "
            "historicamente apontam para uma possível queda."
        )
    else:
        resposta_curta = "Sem sinal claro no momento"
        cor_classe = "alert-banner-neutral"
        emoji = ""
        explicacao = (
            f"Os indicadores de <strong>{nome_acao}</strong> estão equilibrados. "
            "Não há sinal forte de alta nem de baixa nas últimas semanas. "
            "Pode ser prudente aguardar movimentos mais claros antes de decidir."
        )

    st.markdown(f"""
    <div class="{cor_classe}">
        <div class="alert-banner-title">{emoji} É UM BOM MOMENTO PARA INVESTIR EM {ticker}?</div>
        <div class="alert-banner-signal" style="font-size:28px;">{resposta_curta}</div>
        <div class="alert-banner-info">{explicacao}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── INFORMAÇÕES SIMPLES (3 cards básicos) ────────────────────────────
    col1, col2, col3 = st.columns(3)

    anterior = df_ticker.iloc[-2] if len(df_ticker) > 1 else ultimo
    var_dia = (ultimo["close"] - anterior["close"]) / anterior["close"] * 100

    with col1:
        st.markdown(f"""
        <div style="background:#161a23; border:1px solid #1f2430; border-radius:10px; padding:18px;">
            <div style="font-size:11px; color:#64748b; font-weight:600;">PREÇO DE HOJE</div>
            <div style="font-size:28px; font-weight:800; color:#f1f5f9;">R$ {preco_atual:.2f}</div>
            <div style="font-size:12px; color:{'#16a34a' if var_dia >= 0 else '#dc2626'}; font-weight:600;">
                {'▲' if var_dia >= 0 else '▼'} {var_dia:+.2f}% desde ontem
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Variação do último mês em linguagem clara
        primeiro_30d = df_ticker.iloc[-30] if len(df_ticker) > 30 else df_ticker.iloc[0]
        var_30d = (ultimo["close"] - primeiro_30d["close"]) / primeiro_30d["close"] * 100
        descr_30d = "subiu" if var_30d > 0 else "caiu"
        st.markdown(f"""
        <div style="background:#161a23; border:1px solid #1f2430; border-radius:10px; padding:18px;">
            <div style="font-size:11px; color:#64748b; font-weight:600;">NO ÚLTIMO MÊS</div>
            <div style="font-size:28px; font-weight:800; color:{'#16a34a' if var_30d >= 0 else '#dc2626'};">
                {var_30d:+.2f}%
            </div>
            <div style="font-size:12px; color:#64748b;">
                A ação {descr_30d} cerca de R$ {abs(ultimo['close']-primeiro_30d['close']):.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        descr_prev = "ligeira alta" if var_esperada_pct > 1 else ("ligeira queda" if var_esperada_pct < -1 else "estabilidade")
        st.markdown(f"""
        <div style="background:#161a23; border:1px solid #1f2430; border-radius:10px; padding:18px;">
            <div style="font-size:11px; color:#64748b; font-weight:600;">ESTIMATIVA PARA OS PRÓXIMOS 7 PREGÕES</div>
            <div style="font-size:28px; font-weight:800; color:#f1f5f9;">R$ {preco_previsto:.2f}</div>
            <div style="font-size:12px; color:#64748b;">
                O modelo prevê {descr_prev}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── GRÁFICO SIMPLES ─────────────────────────────────────────────────
    st.markdown("### Como a ação se comportou ao longo do tempo?")

    periodo_label = st.selectbox(
        "Período do gráfico:", list(PERIODOS_GRAFICO.keys()),
        index=1, key="periodo_iniciante",
    )
    fig = _grafico_simples_iniciante(df_ticker, df_previsao, ticker, PERIODOS_GRAFICO[periodo_label])
    st.plotly_chart(fig, use_container_width=True)

    # ── BOX "O QUE ISSO SIGNIFICA?" ──────────────────────────────────────
    st.markdown("### O que isso significa pra mim?")

    if sinal == "compra":
        texto_significa = (
            "**Em linguagem simples:** o sistema observa o histórico de preços e os indicadores "
            "técnicos da ação e identificou padrões que, no passado, antecederam períodos de alta. "
            "Isso **não garante** que a ação vai subir — apenas indica que os sinais atuais são "
            "semelhantes aos de momentos anteriores em que houve valorização.\n\n"
            "**Próximos passos sugeridos:** se você estava considerando investir nessa ação, "
            "este pode ser um momento para estudar a empresa com mais atenção, ver notícias "
            "recentes sobre o setor financeiro, e conversar com seu assessor de investimentos "
            "(se você tiver um) antes de tomar qualquer decisão."
        )
    elif sinal == "venda":
        texto_significa = (
            "**Em linguagem simples:** o sistema identificou nos indicadores técnicos uma "
            "combinação que historicamente precedeu períodos de queda. Isso **não significa** "
            "que a ação vai cair com certeza — apenas que os sinais atuais são parecidos com "
            "os de momentos anteriores em que houve desvalorização.\n\n"
            "**Próximos passos sugeridos:** se você já possui essa ação, pode ser um bom momento "
            "para revisar sua estratégia. Se estava pensando em comprar, talvez valha a pena "
            "esperar sinais mais positivos. Considere conversar com seu assessor de investimentos."
        )
    else:
        texto_significa = (
            "**Em linguagem simples:** os indicadores técnicos não apontam direção clara no momento. "
            "Não há sinais fortes de alta nem de baixa. Isso pode acontecer quando a ação está "
            "em um período de lateralização (preço oscilando dentro de uma faixa estreita) ou "
            "em transição entre tendências.\n\n"
            "**Próximos passos sugeridos:** aguardar é geralmente uma opção razoável quando o "
            "mercado está sem direção clara. Continue acompanhando — os sinais podem mudar "
            "nas próximas semanas conforme novos dados forem incorporados."
        )

    with st.container(border=True):
        st.markdown(texto_significa)

    # ── HISTÓRICO RECENTE EM LINGUAGEM SIMPLES ───────────────────────────
    st.markdown("### O que aconteceu nas últimas semanas?")

    # Pega os últimos 3 sinais que não foram neutros
    sinais_recentes = []
    for i in range(len(df_ticker) - 1, max(0, len(df_ticker) - 90), -1):
        s, _ = obter_sinal(df_ticker.iloc[i])
        if s != "neutro":
            sinais_recentes.append((df_ticker.iloc[i], s))
        if len(sinais_recentes) >= 3:
            break

    if sinais_recentes:
        with st.container(border=True):
            for linha, s in sinais_recentes:
                data_str = linha["data"].strftime("%d/%m/%Y")
                preco = linha["close"]
                if s == "compra":
                    st.markdown(
                        f"**{data_str}** — Em R$ {preco:.2f}, o sistema identificou "
                        "um possível ponto de entrada (sinal de compra)."
                    )
                else:
                    st.markdown(
                        f"**{data_str}** — Em R$ {preco:.2f}, o sistema identificou "
                        "um possível ponto de saída (sinal de venda)."
                    )
    else:
        st.info("Nas últimas semanas o sistema não identificou sinais claros.")

    # ── AVISO IMPORTANTE ────────────────────────────────────────────────
    st.markdown("---")
    st.warning(
        "**Importante:** este sistema é uma ferramenta de **apoio à decisão**, "
        "baseada em análise técnica histórica e Machine Learning. **Não constitui "
        "recomendação de investimento**. Toda decisão de investir envolve riscos. "
        "Considere consultar um profissional credenciado (CVM) antes de tomar "
        "qualquer decisão financeira."
    )

    # ── CHAMADA PARA MODO EXPERT ────────────────────────────────────────
    with st.expander("Quer ver a análise técnica completa?"):
        st.markdown(
            "O **Painel Avançado** mostra todos os indicadores técnicos brutos (RSI, MACD, "
            "Bandas de Bollinger, Médias Móveis), as métricas dos modelos de "
            "Machine Learning (F1-Score, Precision, Recall), a faixa de confiança "
            "estatística da previsão e as três projeções (cenário otimista, "
            "esperado e pessimista). É indicado para quem já tem familiaridade "
            "com análise técnica de ativos.\n\n"
            "Para acessar, clique em **Painel Avançado** na barra lateral à esquerda."
        )


def _grafico_simples_iniciante(df_ticker: pd.DataFrame, df_previsao: pd.DataFrame, ticker: str,
                               dias: Optional[int] = None) -> go.Figure:
    """Gráfico simplificado para o modo iniciante — sem indicadores técnicos sobrepostos."""
    df_recente = df_ticker if dias is None else df_ticker.tail(dias)
    fig = go.Figure()

    # Preço histórico
    fig.add_trace(go.Scatter(
        x=df_recente["data"], y=df_recente["close"],
        mode="lines", name="Preço da ação",
        line=dict(color="#1e40af", width=3),
        hovertemplate="%{x|%d/%m/%Y}<br>R$ %{y:.2f}<extra></extra>",
    ))

    # Sinais (apenas alguns, pra não poluir)
    sinais = df_recente.apply(lambda r: obter_sinal(r)[0], axis=1)
    # Pega 1 sinal a cada 10 pra ficar limpo
    indices_compra = df_recente.index[sinais == "compra"][::8]
    indices_venda = df_recente.index[sinais == "venda"][::8]

    if len(indices_compra) > 0:
        fig.add_trace(go.Scatter(
            x=df_recente.loc[indices_compra, "data"],
            y=df_recente.loc[indices_compra, "close"],
            mode="markers", name="Sistema sugeriu COMPRAR",
            marker=dict(symbol="triangle-up", size=18, color="#16a34a",
                       line=dict(color="white", width=2)),
            hovertemplate="%{x|%d/%m/%Y}<br>R$ %{y:.2f}<br><b>Sugestão: COMPRAR</b><extra></extra>",
        ))
    if len(indices_venda) > 0:
        fig.add_trace(go.Scatter(
            x=df_recente.loc[indices_venda, "data"],
            y=df_recente.loc[indices_venda, "close"],
            mode="markers", name="Sistema sugeriu VENDER",
            marker=dict(symbol="triangle-down", size=18, color="#dc2626",
                       line=dict(color="white", width=2)),
            hovertemplate="%{x|%d/%m/%Y}<br>R$ %{y:.2f}<br><b>Sugestão: VENDER</b><extra></extra>",
        ))

    # Estimativa futura (sem faixa de confiança técnica)
    if not df_previsao.empty:
        fig.add_trace(go.Scatter(
            x=df_previsao["data"], y=df_previsao["previsao"],
            mode="lines+markers", name="Estimativa para os próximos dias",
            line=dict(color="#f59e0b", width=3, dash="dash"),
            marker=dict(size=8),
            hovertemplate="%{x|%d/%m/%Y}<br>Estimado: R$ %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="", yaxis_title="Preço em reais (R$)",
        hovermode="x unified", height=420,
        paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
        font=dict(color="#cbd5e1"),
        xaxis=dict(gridcolor="#1f2430", color="#94a3b8"),
        yaxis=dict(gridcolor="#1f2430", color="#94a3b8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def tela_analise_expert(df: pd.DataFrame, modelos: dict) -> None:
    # Seleção do ativo
    col_t, col_d = st.columns([3, 1])
    with col_t:
        ticker = st.selectbox(
            "Selecione a ação:",
            options=list(TICKERS_INFO.keys()),
            format_func=lambda t: f"{t} — {TICKERS_INFO[t]['nome']}",
            key="ticker_analise",
        )
    with col_d:
        dias_previsao = st.slider("Dias de previsão:", 1, 15, 7)

    df_ticker = df[df["ticker"] == ticker].copy().reset_index(drop=True)
    if df_ticker.empty:
        st.warning(f"Sem dados para {ticker}.")
        return

    ultimo = df_ticker.iloc[-1]
    sinal, confianca = obter_sinal(ultimo)
    df_previsao = previsao_lstm_demo(df_ticker, dias_previsao)
    preco_previsto = float(df_previsao["previsao"].iloc[0]) if not df_previsao.empty else float(ultimo["close"])

    st.markdown(f'<div class="page-title">Análise Detalhada — {ticker}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">{TICKERS_INFO[ticker]["nome"]} • '
        f'Última atualização: {ultimo["data"].strftime("%d/%m/%Y")}</div>',
        unsafe_allow_html=True,
    )

    # Banner principal
    banner_alerta(sinal, ticker, float(ultimo["close"]), preco_previsto, confianca)

    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    anterior = df_ticker.iloc[-2] if len(df_ticker) > 1 else ultimo
    var_dia = (ultimo["close"] - anterior["close"]) / anterior["close"] * 100
    primeiro_30d = df_ticker.iloc[-30] if len(df_ticker) > 30 else df_ticker.iloc[0]
    var_30d = (ultimo["close"] - primeiro_30d["close"]) / primeiro_30d["close"] * 100

    col1.metric("Preço atual", f"R$ {ultimo['close']:.2f}", f"{var_dia:+.2f}%")
    col2.metric("Variação 30d", f"{var_30d:+.2f}%")
    col3.metric("Volume hoje", f"{ultimo['volume']/1e6:.1f}M")
    col4.metric("Previsão", f"R$ {preco_previsto:.2f}", f"{(preco_previsto-ultimo['close'])/ultimo['close']*100:+.2f}%")

    st.markdown("### Indicadores técnicos")

    # Cards de indicadores
    rsi_val = ultimo.get("rsi_14", np.nan)
    macd_val = ultimo.get("macd_hist", np.nan)

    rsi_status = "neutro"
    rsi_desc = "Zona neutra"
    if not pd.isna(rsi_val):
        if rsi_val > 70:
            rsi_status = "baixa"; rsi_desc = "Sobrecomprado"
        elif rsi_val < 30:
            rsi_status = "alta"; rsi_desc = "Sobrevendido"
        else:
            rsi_desc = "Zona neutra"

    macd_status = "neutro"
    macd_desc = "Sem sinal claro"
    if not pd.isna(macd_val):
        if macd_val > 0:
            macd_status = "alta"; macd_desc = "Tendência altista"
        else:
            macd_status = "baixa"; macd_desc = "Tendência baixista"

    sma9_val = ultimo.get("sma_9", ultimo["close"])
    sma21_val = ultimo.get("sma_21", ultimo["close"])
    sma_status = "alta" if ultimo["close"] > sma21_val else "baixa"
    sma_desc = f"Preço {'acima' if ultimo['close'] > sma21_val else 'abaixo'} da SMA 21"

    cols_ind = st.columns(4)
    with cols_ind[0]:
        st.markdown(card_indicador(
            "RSI (14)",
            f"{rsi_val:.1f}" if not pd.isna(rsi_val) else "—",
            rsi_desc, rsi_status
        ), unsafe_allow_html=True)
    with cols_ind[1]:
        st.markdown(card_indicador(
            "MACD Histograma",
            f"{macd_val:+.3f}" if not pd.isna(macd_val) else "—",
            macd_desc, macd_status
        ), unsafe_allow_html=True)
    with cols_ind[2]:
        st.markdown(card_indicador(
            "SMA 9",
            f"R$ {sma9_val:.2f}",
            f"Diferença: {(ultimo['close']-sma9_val):+.2f}",
            sma_status
        ), unsafe_allow_html=True)
    with cols_ind[3]:
        st.markdown(card_indicador(
            "SMA 21",
            f"R$ {sma21_val:.2f}",
            sma_desc, sma_status
        ), unsafe_allow_html=True)

    # Gráfico principal
    periodo_label = st.selectbox(
        "Período do gráfico:", list(PERIODOS_GRAFICO.keys()),
        index=1, key="periodo_avancado",
    )
    st.plotly_chart(
        grafico_principal(df_ticker, df_previsao, ticker, PERIODOS_GRAFICO[periodo_label]),
        use_container_width=True,
    )

    # Cenários de predição
    st.markdown("### Cenários de predição (próximos pregões)")
    col_o, col_e, col_p = st.columns(3)
    preco_ult = float(ultimo["close"])
    preco_med = float(df_previsao["previsao"].iloc[-1]) if not df_previsao.empty else preco_ult
    preco_otim = preco_med * 1.04
    preco_pess = preco_med * 0.96

    with col_o:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.08); border-left:3px solid #10b981; padding:14px; border-radius:8px;">
            <div style="font-size:11px; color:#15803d; font-weight:700;">CENÁRIO OTIMISTA</div>
            <div style="font-size:22px; font-weight:800; color:#f1f5f9; margin:6px 0;">R$ {preco_otim:.2f}</div>
            <div style="font-size:12px; color:#15803d; font-weight:700;">▲ {(preco_otim-preco_ult)/preco_ult*100:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_e:
        st.markdown(f"""
        <div style="background:rgba(59,130,246,0.08); border-left:3px solid #3b82f6; padding:14px; border-radius:8px;">
            <div style="font-size:11px; color:#1e40af; font-weight:700;">CENÁRIO ESPERADO</div>
            <div style="font-size:22px; font-weight:800; color:#f1f5f9; margin:6px 0;">R$ {preco_med:.2f}</div>
            <div style="font-size:12px; color:#1e40af; font-weight:700;">{(preco_med-preco_ult)/preco_ult*100:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p:
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.08); border-left:3px solid #ef4444; padding:14px; border-radius:8px;">
            <div style="font-size:11px; color:#dc2626; font-weight:700;">CENÁRIO PESSIMISTA</div>
            <div style="font-size:22px; font-weight:800; color:#f1f5f9; margin:6px 0;">R$ {preco_pess:.2f}</div>
            <div style="font-size:12px; color:#dc2626; font-weight:700;">▼ {(preco_pess-preco_ult)/preco_ult*100:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)


def tela_alertas(df: pd.DataFrame) -> None:
    st.markdown('<div class="page-title">Centro de Alertas</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Histórico e notificações de sinais — entrega prevista para o TC II</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="background:#161a23;
                border:1px solid #1f2430; border-radius:12px; padding:48px 32px;
                text-align:center; margin-top:8px;">
        <span class="economia-badge">EM DESENVOLVIMENTO · TC II</span>
        <h2 style="color:#f1f5f9; margin-top:16px; font-size:24px;">Central de Alertas em construção</h2>
        <p style="color:#cbd5e1; line-height:1.7; margin-top:8px; font-size:14px;">
            O histórico de alertas e o envio de notificações de compra e venda
            serão implementados na entrega final do TCC (TC II).
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# APLICAÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Econom-IA",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    aplicar_css()

    # Estado de navegação por seções
    if "pagina_ativa" not in st.session_state:
        st.session_state.pagina_ativa = "inicio"

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom:24px;">
            <div style="font-weight:700; color:#f1f5f9; font-size:20px;">Econom-IA</div>
            <div style="font-size:11px; color:#64748b;">TCC - UNOESC 2026</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Seção INÍCIO ─────────────────────────────────────
        st.markdown('<div class="sidebar-section-title">INÍCIO</div>', unsafe_allow_html=True)
        if st.button("Visão Geral", key="btn_inicio",
                     use_container_width=True,
                     type="primary" if st.session_state.pagina_ativa == "inicio" else "secondary"):
            st.session_state.pagina_ativa = "inicio"
            st.rerun()

        # ── Seção ANÁLISE ────────────────────────────────────
        st.markdown('<div class="sidebar-section-title">ANÁLISE</div>', unsafe_allow_html=True)
        if st.button("Painel Iniciante", key="btn_iniciante",
                     use_container_width=True,
                     type="primary" if st.session_state.pagina_ativa == "iniciante" else "secondary"):
            st.session_state.pagina_ativa = "iniciante"
            st.rerun()
        if st.button("Painel Avançado", key="btn_avancado",
                     use_container_width=True,
                     type="primary" if st.session_state.pagina_ativa == "avancado" else "secondary"):
            st.session_state.pagina_ativa = "avancado"
            st.rerun()

        # ── Seção FERRAMENTAS ────────────────────────────────
        # NOTA: a aba EconomIA (assistente conversacional) foi removida nesta
        # etapa e será adicionada no TCC II. A função tela_economia_funcional
        # permanece no código para retomada futura.
        st.markdown('<div class="sidebar-section-title">FERRAMENTAS</div>', unsafe_allow_html=True)
        if st.button("Previsões & Macro", key="btn_previsoes",
                     use_container_width=True,
                     type="primary" if st.session_state.pagina_ativa == "previsoes" else "secondary"):
            st.session_state.pagina_ativa = "previsoes"
            st.rerun()
        if st.button("Alertas", key="btn_alertas",
                     use_container_width=True,
                     type="primary" if st.session_state.pagina_ativa == "alertas" else "secondary"):
            st.session_state.pagina_ativa = "alertas"
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);
                        padding:10px; border-radius:8px;">
                <div style="font-size:12px; font-weight:700; color:#fbbf24;">Aviso importante</div>
                <div style="font-size:11px; color:#cbd5e1; margin-top:4px; line-height:1.5;">
                    Esta plataforma é educacional. Não constitui recomendação de investimento. Consulte um assessor.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-top:16px; padding-top:12px; border-top:1px solid #1f2430;
                        font-size:11px; color:#64748b;">
                <strong style="color:#94a3b8;">TCC 2026</strong> · UNOESC<br>
                Kauan Cipriani · Vitor Konzen
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Carregamento de dados ─────────────────────────────────────────────
    df, modo = carregar_dados()

    if df is None:
        st.error(
            "Nenhum arquivo de dados encontrado. Execute primeiro o script "
            "`coleta_dados.py` (Fase 1) para gerar o dataset."
        )
        st.code("python src/01_coleta_dados.py", language="bash")
        return

    modelos = carregar_modelos()
    metricas = carregar_metricas()

    # Gera os sinais de compra/venda/neutro com o Random Forest treinado
    df = adicionar_sinais_modelo(df, modelos)

    # Status do modelo no rodapé da sidebar
    with st.sidebar:
        if metricas:
            st.markdown(
                f"""
                <div style="background:rgba(16,185,129,0.1); border:1px solid #1f2430;
                            padding:10px; border-radius:8px; margin-top:8px;">
                    <div style="font-size:11px; font-weight:700; color:#10b981;">Modelo treinado</div>
                    <div style="font-size:10px; color:#cbd5e1; margin-top:4px;">
                        F1 OOS: {metricas['resultados_oos']['f1_macro']:.3f} · {metricas['info_treinamento']['data_treinamento'][:10]}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

        # Status / atualização dos dados de mercado
        ultima_data = df["data"].max()
        rotulo_modo = {
            "online": "Dados ao vivo (Yahoo Finance)",
            "features": "Dados locais (offline)",
            "ohlcv": "Dados locais (offline)",
        }.get(modo, "Dados de mercado")
        st.markdown(
            f"""
            <div style="background:rgba(59,130,246,0.1); border:1px solid #1f2430;
                        padding:10px; border-radius:8px; margin-top:8px;">
                <div style="font-size:11px; font-weight:700; color:#60a5fa;">{rotulo_modo}</div>
                <div style="font-size:10px; color:#cbd5e1; margin-top:4px;">
                    Última cotação: {ultima_data:%d/%m/%Y} · atualiza a cada 15 min
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("Atualizar agora", key="btn_refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Roteamento ────────────────────────────────────────────────────────
    pagina = st.session_state.pagina_ativa

    if pagina == "inicio":
        tela_inicio(df)
    elif pagina == "iniciante":
        tela_analise_iniciante(df, modelos)
    elif pagina == "avancado":
        tela_analise_expert(df, modelos)
    elif pagina == "previsoes":
        tela_previsoes_macro()
    elif pagina == "alertas":
        tela_alertas(df)


if __name__ == "__main__":
    main()
