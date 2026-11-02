"""
Coleta das cotações diárias no Yahoo Finance.

Baixa o OHLCV ajustado das quatro ações monitoradas e grava o arquivo
dados/b3_financeiro_ohlcv.parquet, usado pelo treinamento e como cópia de
segurança da interface quando não há conexão com a internet.

Como executar (dentro da pasta projeto_tcc):
    python coletar_dados.py                      # de 2019-01-01 até hoje
    python coletar_dados.py --fim 2025-09-30     # até uma data específica
"""

import argparse
import sys

from econom_ia.dados import ColetorYahoo, ErroColeta, LimpadorDados, RepositorioDados


def main() -> None:
    parser = argparse.ArgumentParser(description="Atualiza a base de cotações pelo Yahoo Finance.")
    parser.add_argument("--inicio", default="2019-01-01", help="data inicial (AAAA-MM-DD)")
    parser.add_argument("--fim", default=None, help="data final inclusiva (AAAA-MM-DD)")
    argumentos = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        dados = ColetorYahoo(inicio=argumentos.inicio, fim=argumentos.fim).baixar()
    except ErroColeta as erro:
        sys.exit(f"Falha na coleta: {erro}")

    RepositorioDados().salvar_ohlcv(dados)
    _, relatorio = LimpadorDados().limpar(dados)
    print(f"{len(dados)} registros gravados em dados/b3_financeiro_ohlcv.parquet")
    print(dados.groupby("ticker")["data"].agg(["min", "max", "count"]).to_string())
    print(f"Dias sem negociação reportados pela fonte: {relatorio.sem_negociacao_removidos}")
    print(f"Retornos extremos marcados como outliers: {relatorio.outliers_marcados}")


if __name__ == "__main__":
    main()
