"""
Treinamento e avaliação dos modelos do Econom-IA.

Executa o pipeline completo (PipelineTreinamento): limpeza dos dados,
indicadores técnicos, reavaliação do limiar, comparação de variáveis, ajuste
de hiperparâmetros do Random Forest, SVM, MLP e LSTM, avaliação no teste,
na ação out-of-sample (SANB11) e no período prospectivo, e gravação dos
modelos e do arquivo modelos/metricas.json lido pela interface.

Como executar (dentro da pasta projeto_tcc):
    python treinar_modelos.py            # treinamento completo (cerca de 5 a 10 min)
    python treinar_modelos.py --rapido   # verificação rápida do pipeline
"""

import argparse
import logging
import sys

from econom_ia.treinamento import PipelineTreinamento


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina e avalia os modelos do Econom-IA.")
    parser.add_argument("--rapido", action="store_true",
                        help="usa grades e épocas reduzidas (apenas para testar o pipeline)")
    argumentos = parser.parse_args()

    # Evita erros de acentuação no terminal do Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    PipelineTreinamento(modo_rapido=argumentos.rapido).executar()


if __name__ == "__main__":
    main()
