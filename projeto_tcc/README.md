# Econom-IA

**Sistema de Apoio à Decisão para Predição de Tendências de Ativos da B3 utilizando Machine Learning**

Trabalho de Conclusão de Curso em Ciência da Computação, Unoesc São Miguel do Oeste, 2026.
Alunos: Kauan Amélio Cipriani e Vitor Hugo Konzen. Orientador: Vinícius Almeida dos Santos.

O sistema acompanha quatro ações do setor bancário (ITUB4, BBDC4, BBAS3 e SANB11), calcula indicadores
técnicos, gera sinais de **compra, venda ou neutro** com quatro modelos de Machine Learning (Random Forest,
SVM, MLP e LSTM), prevê o preço dos próximos 1 a 15 pregões com uma rede LSTM e apresenta tudo em uma
interface web com linguagem para iniciantes e para usuários avançados, além de indicadores
macroeconômicos do Brasil e dos Estados Unidos.

> Ferramenta acadêmica de apoio à decisão. Não constitui recomendação de investimento.

---

## Como executar

### 1. Pré-requisitos
- **Python 3.11 ou superior** ([python.org/downloads](https://www.python.org/downloads/))

### 2. Crie e ative um ambiente virtual (dentro da pasta `projeto_tcc`)

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt --upgrade
```
A instalação leva alguns minutos (o PyTorch é o pacote maior).

### 4. Abra a interface
```bash
streamlit run app.py
```
O navegador abre em `http://localhost:8501`. Para encerrar, use `Ctrl + C` no terminal.

Os modelos já treinados e a base de cotações acompanham o projeto, então a interface funciona logo após a
instalação. Com internet, as cotações e os indicadores são atualizados automaticamente; sem internet, o
sistema usa a base local e as últimas cópias salvas das APIs.

---

## Atualizar os dados e retreinar os modelos

```bash
python coletar_dados.py        # baixa as cotações de 2019 até hoje (Yahoo Finance)
python treinar_modelos.py      # experimentos, ajuste de hiperparâmetros e avaliação (cerca de 5 a 10 min)
```
O treinamento grava os modelos em `modelos/*.pkl` e todas as métricas em `modelos/metricas.json`, lidos
pela tela **Desempenho dos Modelos**. Depois de retreinar com a interface aberta, clique em
**Atualizar dados agora** na barra lateral.

## Testes automatizados

```bash
python -m pytest
```
São 45 testes: indicadores (inclusive a ausência de vazamento de informação futura), rotulagem, validação
temporal, métricas, modelos (treino, previsão e persistência), integrações com as APIs (com respostas
simuladas), serviços, as seis telas da interface e a personalização das telas.

---

## Estrutura do projeto

```
projeto_tcc/
├── app.py                  ponto de entrada da interface (streamlit run app.py)
├── coletar_dados.py        coleta das cotações no Yahoo Finance
├── treinar_modelos.py      treinamento e avaliação dos modelos
├── requirements.txt        dependências
├── dados/                  base de cotações (Parquet) e cópias locais das APIs
├── modelos/                modelos treinados (.pkl) e métricas (metricas.json)
├── tests/                  testes automatizados (pytest)
└── econom_ia/              pacote com todo o código, organizado em camadas
    ├── config.py           parâmetros do estudo (ativos, períodos, horizonte, limiares)
    ├── dados/              coleta, limpeza e persistência (ColetorYahoo, LimpadorDados, RepositorioDados)
    ├── features/           indicadores técnicos e variável-alvo (CalculadoraIndicadores, Rotulador)
    ├── modelos/            ModeloClassificacao (classe abstrata) e as subclasses RF, SVM, MLP e LSTM
    ├── avaliacao/          validação temporal com expurgo, métricas, intervalos de confiança e backtest
    ├── treinamento/        PipelineTreinamento: orquestra todos os experimentos
    ├── integracoes/        clientes das APIs (Banco Central, Focus, FRED, Polymarket, Yahoo, Google Notícias)
    ├── servicos/           regras usadas pela interface (sinais, previsão, alertas, cotações)
    └── interface/          aplicação Streamlit: tema, componentes, gráficos e as seis telas
```

O código segue a Programação Orientada a Objetos: cada responsabilidade fica em uma classe, os modelos
compartilham a interface de `ModeloClassificacao` (polimorfismo), as integrações herdam de `ClienteHTTP`
(tempo-limite e cache em disco) e as telas herdam de `Tela`.

## Telas

O menu fica na barra lateral, agrupado em Início, Análise e Ferramentas, e cada tela tem endereço próprio (o botão
"voltar" do navegador funciona). A barra pode ser recolhida pelo botão « e reaberta pelo botão "Menu".
Em cada tela, o botão **Personalizar tela** mostra ou esconde os blocos opcionais (indicadores técnicos,
gráficos, manchetes etc.); a resposta principal e o aviso legal ficam sempre visíveis. A situação dos modelos e
dos dados, a opção de usar só a base local e o botão de atualização ficam na barra lateral, abaixo do menu.

| Tela | Conteúdo |
|---|---|
| Visão Geral | humor do setor, Ibovespa, dólar, Selic, IPCA, desempenho dos bancos, manchetes da semana e sinais do dia |
| Painel Iniciante | resposta simples ao "é um bom momento?", estimativa de preço, histórico recente de alertas e seus acertos |
| Painel Avançado | sinal de cada modelo com probabilidades, indicadores técnicos, gráfico com alertas e previsão da LSTM, cenários e o que mais pesou no sinal |
| Previsões & Macro | Selic, IPCA, juro real, PTAX, juros dos EUA, expectativas do Relatório Focus e probabilidades do Polymarket |
| Centro de Alertas | histórico filtrável dos alertas de compra e venda, taxa de acerto por ação e exportação em CSV |
| Desempenho dos Modelos | comparação dos modelos com referências e intervalos de confiança, experimentos, LSTM e simulação de carteira |

## Fontes de dados

| Fonte | Uso |
|---|---|
| Yahoo Finance (yfinance) | cotações diárias ajustadas das quatro ações, Ibovespa e dólar |
| Banco Central do Brasil, API SGS | meta Selic (432), IPCA em 12 meses (13522) e dólar PTAX (1) |
| Banco Central do Brasil, API Olinda | expectativas do Relatório Focus para a Selic em cada reunião do Copom |
| FRED (Federal Reserve Bank of St. Louis) | Fed Funds (DFF) e juros do Tesouro americano (DGS3MO, DGS2, DGS10) |
| Polymarket (Gamma API) | probabilidades para a próxima decisão do Copom |
| Google Notícias (RSS) | manchetes da semana sobre os quatro bancos |

Os dados macroeconômicos e as manchetes são apenas contextuais: os modelos usam somente os indicadores
técnicos calculados a partir das cotações.

**Polymarket no Brasil:** desde abril de 2026 o acesso à plataforma é bloqueado no país por determinação da
Secretaria de Prêmios e Apostas do Ministério da Fazenda. Em redes brasileiras a consulta costuma falhar; o
sistema detecta a falha, não trava e mostra as expectativas do Relatório Focus, que cumprem o mesmo papel
com uma fonte oficial.

## Metodologia (resumo)

- Cotações reais de jan/2019 a set/2026; período de estudo até set/2025 e período prospectivo a partir de
  out/2025, nunca usado no treinamento.
- Variável-alvo: retorno em 5 pregões acima do limiar (compra), abaixo do limiar negativo (venda) ou entre
  eles (neutro). O limiar de ±1,5% do TC I foi reavaliado com limiares fixos e adaptativos à volatilidade.
- Variáveis: as 8 do TC I e 9 novas (volume, volatilidade relativa, ATR e retornos passados normalizados). A
  comparação por validação cruzada escolheu o conjunto das 9 novas, usado pelo Random Forest, SVM e MLP.
- Treino em ITUB4, BBDC4 e BBAS3; SANB11 reservada para avaliar a generalização (out-of-sample).
- Validação cruzada temporal com expurgo de 5 pregões; busca de hiperparâmetros em grade; métricas F1-Score
  macro e MCC com intervalos de confiança por bootstrap em blocos; comparação com classificadores de
  referência; simulação de carteira com custos.

Os resultados detalhados estão na tela **Desempenho dos Modelos** e no artigo do TCC.
