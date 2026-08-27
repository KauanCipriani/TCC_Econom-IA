# Econom-IA — Sistema de Apoio à Decisão para Ações da B3

**TCC — Ciência da Computação | UNOESC São Miguel do Oeste — 2026**

Sistema baseado em Machine Learning (Random Forest) para predição de tendências
de ações do setor financeiro da B3, com interface web em Streamlit.

**Alunos:** Kauan Amélio Cipriani • Vitor Hugo Konzen
**Orientador:** Vinícius Almeida dos Santos

---

## Como rodar (passo a passo)

### 1. Pré-requisitos
- **Python 3.10 ou superior** ([baixar aqui](https://www.python.org/downloads/))

### 2. (Recomendado) Crie um ambiente virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Rode o sistema
```bash
streamlit run app.py
```
Abre automaticamente em `http://localhost:8501`. Para parar: `Ctrl + C`.

---

## Estrutura dos arquivos

```
projeto_tcc/
├── app.py                       ← Interface Streamlit (todas as telas)
├── treinar_modelos.py           ← Pipeline de treinamento do Random Forest
├── requirements.txt             ← Dependências Python
├── .streamlit/
│   └── config.toml              ← Tema escuro
├── dados/
│   ├── b3_financeiro_ohlcv.parquet      ← OHLCV bruto (jan/2019–set/2025)
│   └── b3_financeiro_features.parquet   ← Dataset com indicadores
└── modelos/
    ├── random_forest.pkl        ← Modelo treinado
    └── metricas.json            ← Métricas reais do treinamento
```

---

## O que o sistema faz

Os **sinais de compra/venda/neutralidade** são gerados pelo **modelo Random
Forest** treinado, a partir dos indicadores técnicos de cada ação. Os preços
são obtidos **ao vivo** do Yahoo Finance (via `yfinance`); sem conexão, o app
usa os dados locais (Parquet) como cópia de segurança.

### Telas (barra lateral)

**INÍCIO**
- **Visão Geral** — boas-vindas, 4 cards macro ao vivo (IBOVESPA e USD/BRL via
  Yahoo Finance; SELIC e IPCA via API do Banco Central) e resumo dos sinais.
  *(Área de notícias prevista para o TC II.)*

**ANÁLISE**
- **Painel Iniciante** — linguagem simples, 3 cards e gráfico com os sinais.
- **Painel Avançado** — indicadores técnicos (RSI, MACD, Bollinger, médias
  móveis), gráfico interativo e seletor de período (mês / 6 meses / ano / desde
  2019; padrão: 6 meses).

**FERRAMENTAS** *(previstas para o TC II)*
- **EconomIA** — assistente conversacional.
- **Previsões & Macro** — Selic/IPCA/Fed/USD, Treasury Yields e Polymarket.
  Dados apenas contextuais — não influenciam os sinais do modelo.
- **Alertas** — histórico de sinais.

Ativos monitorados: **ITUB4, BBDC4, BBAS3 e SANB11**.

---

## Retreinar o modelo (opcional)

```bash
python treinar_modelos.py
```

> No Windows, se houver erro de codificação (`UnicodeEncodeError`), rode com
> UTF-8: no PowerShell `\$env:PYTHONUTF8=1; python treinar_modelos.py`; no CMD
> `set PYTHONUTF8=1 && python treinar_modelos.py`.

O script carrega o OHLCV, calcula os indicadores, cria o alvo de classificação
(limiar ±1,5%, horizonte de 5 pregões), separa SANB11 como out-of-sample,
divide o restante 70/30 por ordem temporal, treina o Random Forest com
TimeSeriesSplit (5 folds) e salva o modelo e o `metricas.json`, lido
automaticamente pela interface.

---

## Resultados do treinamento (Random Forest)

| Métrica (macro) | Conjunto de teste | Out-of-sample (SANB11) |
|---|---|---|
| F1-Score | 0,3238 | 0,3596 |
| Precision | 0,3263 | 0,3601 |
| Recall | 0,3234 | 0,3600 |
| Acurácia | 0,3254 | 0,3678 |

Validação cruzada (TimeSeriesSplit, 5 folds): F1-macro médio **0,3307** (±0,0299).

**Features mais importantes:** `macd_hist_z` (15,99%), `sma_9_ratio` (13,35%),
`sma_50_ratio` (12,87%). Valores próximos de um classificador aleatório
(~0,333), esperado nesta fase (TC I); SVM e LSTM ficam para o TC II.

---

## Tecnologias

Python • Streamlit • Plotly • scikit-learn • pandas • NumPy • joblib •
PyArrow • yfinance • API do Banco Central do Brasil (SGS).

---

## Aviso legal

Este sistema é uma **ferramenta de apoio à decisão de natureza acadêmica**.
**Não constitui recomendação de investimento.** Consulte um profissional
credenciado pela CVM antes de tomar decisões financeiras.