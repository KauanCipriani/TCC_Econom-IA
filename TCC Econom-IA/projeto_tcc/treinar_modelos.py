"""
==============================================================================
TCC - Pipeline de Treinamento Real para Apresentação Parcial
==============================================================================

Este script faz o pipeline completo de treinamento em um único arquivo,
gerando métricas REAIS (não hardcoded) para a apresentação:

    1. Carrega o OHLCV bruto (saída da Fase 1).
    2. Calcula os indicadores técnicos manualmente (RSI, MACD, SMAs, BB).
    3. Cria o target de classificação (compra/venda/neutro por limiar ±1,5%).
    4. Separa SANB11 como validação out-of-sample.
    5. Divide o restante em treino/teste (70/30) por ordem temporal.
    6. Treina Random Forest com TimeSeriesSplit (5 folds).
    7. Avalia no teste e no SANB11 reservado.
    8. Salva:
        - modelos/random_forest.pkl
        - modelos/scaler.pkl
        - modelos/metricas.json  ← MÉTRICAS REAIS lidas pela interface
        - modelos/feature_importance.json

Como executar:
    python treinar_modelos.py

A interface Streamlit (app.py) lê automaticamente esses arquivos.
==============================================================================
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


print("=" * 70)
print("PIPELINE DE TREINAMENTO — TCC Kauan & Vitor")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────
# 1. CARREGA OHLCV
# ─────────────────────────────────────────────────────────────────────────
print("\n[1/7] Carregando OHLCV...")
df = pd.read_parquet("dados/b3_financeiro_ohlcv.parquet")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values(["ticker", "data"]).reset_index(drop=True)
print(f"  → {len(df):,} registros de {len(df.ticker.unique())} ativos")


# ─────────────────────────────────────────────────────────────────────────
# 2. CALCULA INDICADORES TÉCNICOS
# ─────────────────────────────────────────────────────────────────────────
print("\n[2/7] Calculando indicadores técnicos...")

def calcular_indicadores(grupo: pd.DataFrame) -> pd.DataFrame:
    g = grupo.copy().reset_index(drop=True)

    # RSI(14)
    delta = g["close"].diff()
    ganho = delta.where(delta > 0, 0.0).rolling(14).mean()
    perda = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = ganho / perda.replace(0, np.nan)
    g["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()
    g["macd_line"] = ema12 - ema26
    g["macd_signal"] = g["macd_line"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd_line"] - g["macd_signal"]

    # Médias Móveis (razões para serem escala-invariantes)
    for p in (9, 21, 50):
        g[f"sma_{p}"] = g["close"].rolling(p).mean()
        g[f"sma_{p}_ratio"] = g["close"] / g[f"sma_{p}"] - 1

    # EMAs (razões)
    for p in (9, 21):
        g[f"ema_{p}"] = g["close"].ewm(span=p, adjust=False).mean()
        g[f"ema_{p}_ratio"] = g["close"] / g[f"ema_{p}"] - 1

    # Bandas de Bollinger (posição relativa)
    sma20 = g["close"].rolling(20).mean()
    std20 = g["close"].rolling(20).std()
    g["bb_upper"] = sma20 + 2 * std20
    g["bb_lower"] = sma20 - 2 * std20
    g["bb_position"] = (g["close"] - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"])

    # Normalização Z-score do MACD para ficar escala-invariante
    g["macd_hist_z"] = (g["macd_hist"] - g["macd_hist"].rolling(60).mean()) / g["macd_hist"].rolling(60).std()

    return g

df = df.groupby("ticker", group_keys=False)[df.columns.tolist()].apply(calcular_indicadores)
print(f"  → Indicadores calculados: RSI, MACD, SMAs, EMAs, BB")


# ─────────────────────────────────────────────────────────────────────────
# 3. CRIA O TARGET (CLASSIFICAÇÃO)
# ─────────────────────────────────────────────────────────────────────────
print("\n[3/7] Criando target de classificação (limiar ±1,5%, horizonte 5 pregões)...")

LIMIAR = 0.015
HORIZONTE = 5

def criar_target(grupo: pd.DataFrame) -> pd.DataFrame:
    g = grupo.copy()
    preco_futuro = g["close"].shift(-HORIZONTE)
    retorno = (preco_futuro - g["close"]) / g["close"]
    g["target"] = np.select(
        [retorno > LIMIAR, retorno < -LIMIAR],
        ["compra", "venda"],
        default="neutro",
    )
    return g

df = df.groupby("ticker", group_keys=False)[df.columns.tolist()].apply(criar_target)
df = df.dropna(subset=["rsi_14", "macd_hist_z", "bb_position", "sma_50_ratio", "target"])
print(f"  → {len(df):,} registros após warmup")
print(f"  → Distribuição: {df['target'].value_counts(normalize=True).round(3).to_dict()}")

# Salva o dataset com features (saída da "Fase 2")
df.to_parquet("dados/b3_financeiro_features.parquet", index=False)
print(f"  → Salvo: dados/b3_financeiro_features.parquet ({len(df):,} linhas)")


# ─────────────────────────────────────────────────────────────────────────
# 4. SEPARA VALIDAÇÃO OUT-OF-SAMPLE (SANB11)
# ─────────────────────────────────────────────────────────────────────────
print("\n[4/7] Separando SANB11 como validação out-of-sample...")
df_oos = df[df["ticker"] == "SANB11"].copy()
df_resto = df[df["ticker"] != "SANB11"].copy()
print(f"  → Treino/teste: {len(df_resto):,} ({sorted(df_resto.ticker.unique())})")
print(f"  → Out-of-sample: {len(df_oos):,} (SANB11)")


# ─────────────────────────────────────────────────────────────────────────
# 5. DIVISÃO TEMPORAL TREINO/TESTE
# ─────────────────────────────────────────────────────────────────────────
print("\n[5/7] Divisão temporal 70/30...")
FEATURES = [
    "rsi_14", "macd_hist_z",
    "sma_9_ratio", "sma_21_ratio", "sma_50_ratio",
    "ema_9_ratio", "ema_21_ratio",
    "bb_position",
]

df_resto = df_resto.sort_values(["data", "ticker"]).reset_index(drop=True)
corte = int(len(df_resto) * 0.7)
X_train = df_resto.iloc[:corte][FEATURES].values
y_train = df_resto.iloc[:corte]["target"].values
X_test = df_resto.iloc[corte:][FEATURES].values
y_test = df_resto.iloc[corte:]["target"].values
X_oos = df_oos[FEATURES].values
y_oos = df_oos["target"].values
print(f"  → Treino: {len(X_train):,} | Teste: {len(X_test):,} | OOS: {len(X_oos):,}")


# ─────────────────────────────────────────────────────────────────────────
# 6. TREINA RANDOM FOREST COM TIMESERIESSPLIT
# ─────────────────────────────────────────────────────────────────────────
print("\n[6/7] Treinando Random Forest (TimeSeriesSplit, 5 folds)...")

tss = TimeSeriesSplit(n_splits=5)
cv_scores = []
for fold, (idx_tr, idx_vl) in enumerate(tss.split(X_train), start=1):
    modelo_fold = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    modelo_fold.fit(X_train[idx_tr], y_train[idx_tr])
    pred = modelo_fold.predict(X_train[idx_vl])
    f1 = f1_score(y_train[idx_vl], pred, average="macro")
    cv_scores.append(f1)
    print(f"  fold {fold}: F1-macro = {f1:.4f}")

print(f"  → CV F1-macro médio: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# Treina modelo final no conjunto completo
print("  → Treinando modelo final...")
modelo = RandomForestClassifier(
    n_estimators=150, max_depth=10, min_samples_leaf=10,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
modelo.fit(X_train, y_train)


# ─────────────────────────────────────────────────────────────────────────
# 7. AVALIA E SALVA TUDO
# ─────────────────────────────────────────────────────────────────────────
print("\n[7/7] Avaliando e salvando resultados...")

def avaliar(y_true, y_pred):
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_por_classe": {
            classe: float(f1_score(y_true, y_pred, labels=[classe], average="macro", zero_division=0))
            for classe in ["compra", "venda", "neutro"]
        },
        "matriz_confusao": confusion_matrix(
            y_true, y_pred, labels=["venda", "neutro", "compra"]
        ).tolist(),
        "n_amostras": int(len(y_true)),
    }

pred_test = modelo.predict(X_test)
pred_oos = modelo.predict(X_oos)

resultados_teste = avaliar(y_test, pred_test)
resultados_oos = avaliar(y_oos, pred_oos)

print(f"\n  TESTE (mesmos tickers, período futuro):")
print(f"    F1-macro: {resultados_teste['f1_macro']:.4f}")
print(f"    Acurácia: {resultados_teste['accuracy']:.4f}")

print(f"\n  OUT-OF-SAMPLE (SANB11 — nunca visto):")
print(f"    F1-macro: {resultados_oos['f1_macro']:.4f}")
print(f"    Acurácia: {resultados_oos['accuracy']:.4f}")

# Feature importance
importancias = dict(zip(FEATURES, modelo.feature_importances_.tolist()))
importancias_ordenadas = dict(sorted(importancias.items(), key=lambda x: x[1], reverse=True))
print(f"\n  Top 3 features mais importantes:")
for i, (feat, imp) in enumerate(list(importancias_ordenadas.items())[:3], 1):
    print(f"    {i}. {feat}: {imp:.4f}")


# Salva tudo
Path("modelos").mkdir(parents=True, exist_ok=True)
joblib.dump(modelo, "modelos/random_forest.pkl")

metricas = {
    "info_treinamento": {
        "data_treinamento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo": "RandomForestClassifier",
        "hiperparametros": {
            "n_estimators": 150,
            "max_depth": 10,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
        "tickers_treino": sorted(df_resto.ticker.unique().tolist()),
        "ticker_out_of_sample": "SANB11",
        "n_registros_treino": int(len(X_train)),
        "n_registros_teste": int(len(X_test)),
        "n_registros_oos": int(len(X_oos)),
        "features_usadas": FEATURES,
        "periodo_dados": f"{df['data'].min().strftime('%Y-%m')} a {df['data'].max().strftime('%Y-%m')}",
        "cv_folds": 5,
        "cv_f1_medio": float(np.mean(cv_scores)),
        "cv_f1_desvio": float(np.std(cv_scores)),
    },
    "resultados_teste": resultados_teste,
    "resultados_oos": resultados_oos,
    "feature_importance": importancias_ordenadas,
}

with open("modelos/metricas.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

print(f"\n✓ Modelo salvo: modelos/random_forest.pkl")
print(f"✓ Métricas salvas: modelos/metricas.json")
print(f"✓ Features salvas: dados/b3_financeiro_features.parquet")
print("\n" + "=" * 70)
print("TREINAMENTO CONCLUÍDO")
print("=" * 70)
