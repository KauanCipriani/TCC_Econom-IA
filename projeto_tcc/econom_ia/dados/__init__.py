"""Camada de dados: coleta no Yahoo Finance, limpeza e persistência em Parquet."""

from econom_ia.dados.coletor import ColetorYahoo, ErroColeta, LimpadorDados
from econom_ia.dados.repositorio import RepositorioDados

__all__ = ["ColetorYahoo", "ErroColeta", "LimpadorDados", "RepositorioDados"]
