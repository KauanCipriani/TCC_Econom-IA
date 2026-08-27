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
- **Git** (opcional)

### 2. Entre na pasta do app
O código fica na subpasta `projeto_tcc`:
```bash
cd projeto_tcc
```

### 3. (Recomendado) Crie um ambiente virtual

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

### 4. Instale as dependências
```bash
pip install -r requirements.txt
```
A instalação demora alguns minutos (Streamlit, Plotly, scikit-learn, yfinance, etc.).

### 5. Rode o sistema
```bash
streamlit run app.py
```
Abre automaticamente em `http://localhost:8501`. Para parar: `Ctrl + C` no terminal.

---

## Estrutura dos arquivos

```
TCC Econom-IA/
├── README.md
├── requirements.txt
└── projeto_tcc/
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
são obtidos **ao vivo** do Yahoo Finance (via `yfinance`); se não houver
conexão, o app usa os dados locais (Parquet) como cópia de segurança.

### Telas (barra lateral)

**INÍCIO**
- **Visão Geral** — boas-vindas, 4 cards macro atualizados ao vivo (IBOVESPA e
  USD/BRL via Yahoo Finance; SELIC e IPCA via API do Banco Central) e resumo
  dos sinais dos quatro ativos. *(Área de notícias prevista para o TC II.)*

**ANÁLISE**
- **Painel Iniciante** — linguagem simples, 3 cards (preço, variação do mês,
  estimativa) e gráfico com os sinais.
- **Painel Avançado** — indicadores técnicos (RSI, MACD, Bollinger, médias
  móveis), gráfico interativo com sinais e seletor de período (mês / 6 meses /
  ano / desde 2019; padrão: 6 meses).

**FERRAMENTAS** *(previstas para o TC II)*
- **EconomIA** — assistente conversacional.
- **Previsões & Macro** — Selic/IPCA/Fed/USD, Treasury Yields e Polymarket
  (Copom). Dados apenas contextuais — não influenciam os sinais do modelo.
- **Alertas** — histórico de sinais.

Os quatro ativos monitorados: **ITUB4, BBDC4, BBAS3 e SANB11**.

---

## Retreinar o modelo (opcional)

Para regenerar `modelos/random_forest.pkl` e `modelos/metricas.json`:

```bash
cd projeto_tcc
python treinar_modelos.py
```

> No Windows, se aparecer erro de codificação (`UnicodeEncodeError`), rode com
> UTF-8: no PowerShell `\$env:PYTHONUTF8=1; python treinar_modelos.py`; no CMD
> `set PYTHONUTF8=1 && python treinar_modelos.py`.

O script: carrega o OHLCV, calcula os indicadores técnicos, cria o alvo de
classificação (limiar ±1,5%, horizonte de 5 pregões), separa SANB11 como
out-of-sample, divide o restante 70/30 por ordem temporal, treina o Random
Forest com TimeSeriesSplit (5 folds) e salva o modelo e as métricas. A
interface lê o `metricas.json` automaticamente.

---

## Resultados do treinamento (Random Forest)

| Métrica (macro) | Conjunto de teste | Out-of-sample (SANB11) |
|---|---|---|
| F1-Score | 0,3238 | 0,3596 |
| Precision | 0,3263 | 0,3601 |
| Recall | 0,3234 | 0,3600 |
| Acurácia | 0,3254 | 0,3678 |

Validação cruzada (TimeSeriesSplit, 5 folds): F1-macro médio **0,3307** (±0,0299).

**Features mais importantes:**
1. `macd_hist_z` (z-score do histograma MACD): **15,99%**
2. `sma_9_ratio` (razão preço/SMA-9): **13,35%**
3. `sma_50_ratio` (razão preço/SMA-50): **12,87%**

Os valores estão próximos de um classificador aleatório (~0,333), o que é
esperado nesta fase (TC I). Ajuste de hiperparâmetros e a comparação com SVM e
LSTM estão previstos para o TC II.

---

## Problemas comuns

- **"streamlit não é reconhecido"** → ative o ambiente virtual (`(venv)` no
  prompt) ou reinstale as dependências.
- **"ModuleNotFoundError"** → `pip install -r requirements.txt`.
- **Não abre em `localhost:8501`** → tente `http://127.0.0.1:8501` ou verifique
  o firewall.
- **Tema claro em vez de escuro** → confirme que existe `.streamlit/config.toml`.

---

## Tecnologias

Python • Streamlit • Plotly • scikit-learn • pandas • NumPy • joblib •
PyArrow • yfinance • API do Banco Central do Brasil (SGS).

---

## Aviso legal

Este sistema é uma **ferramenta de apoio à decisão de natureza acadêmica**,
baseada em análise técnica histórica e Machine Learning. **Não constitui
recomendação de investimento.** Toda decisão de investir envolve riscos.
Consulte um profissional credenciado pela CVM antes de tomar decisões
financeiras.