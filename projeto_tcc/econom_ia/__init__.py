"""
Econom-IA: Sistema de Apoio à Decisão para Predição de Tendências de Ativos da B3.

Trabalho de Conclusão de Curso (Ciência da Computação, Unoesc São Miguel do Oeste).
Autores: Kauan Amélio Cipriani e Vitor Hugo Konzen.
Orientador: Vinícius Almeida dos Santos.

O pacote está organizado em camadas, cada uma com uma responsabilidade:

    config        parâmetros do experimento, caminhos e cadastro dos ativos
    dados         coleta (Yahoo Finance), limpeza e persistência em Parquet
    features      indicadores técnicos e rotulagem da variável-alvo
    avaliacao     validação temporal, métricas e backtest
    modelos       Random Forest, SVM, MLP e LSTM com interface comum
    treinamento   pipeline completo de experimentos e treinamento
    servicos      sinais, previsões e alertas consumidos pela interface
    integracoes   clientes das APIs externas (BCB, FRED, Polymarket, notícias)
    interface     aplicação Streamlit (telas, componentes e gráficos)
"""

__version__ = "2.0.0"
