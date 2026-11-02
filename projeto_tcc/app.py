"""
Econom-IA: Sistema de Apoio à Decisão para Predição de Tendências de Ativos
da B3 utilizando Machine Learning.

Ponto de entrada da interface web. Execute, dentro da pasta projeto_tcc:

    streamlit run app.py

Toda a lógica fica no pacote econom_ia; este arquivo apenas inicia a aplicação.
"""

from econom_ia.interface.aplicacao import AplicacaoEconomIA

if __name__ == "__main__":
    AplicacaoEconomIA().executar()
