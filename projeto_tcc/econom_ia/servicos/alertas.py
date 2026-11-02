"""
Centro de Alertas: histórico dos sinais emitidos e seu resultado.

Um alerta é registrado quando o sinal de um ativo muda para "compra" ou
"venda" (entrada em um novo estado), com confiança acima do mínimo escolhido
pelo usuário. Depois de "horizonte" pregões compara-se o preço com o do dia
do alerta: o alerta de compra acerta se o preço subiu, e o de venda se caiu.
Isso permite ao usuário fazer a análise retrospectiva do desempenho do
modelo com dados reais, inclusive dos dias mais recentes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from econom_ia.config import ATIVOS


@dataclass
class ResumoAlertas:
    total: int
    avaliados: int
    acertos: int
    aguardando: int
    taxa_acerto: float | None
    retorno_medio_pct: float | None
    por_ativo: pd.DataFrame


class CentralAlertas:
    def __init__(self, horizonte: int = 5, confianca_minima: float = 0.0) -> None:
        self.horizonte = horizonte
        self.confianca_minima = confianca_minima

    def historico(self, dados_sinais: pd.DataFrame, inicio: pd.Timestamp | None = None) -> pd.DataFrame:
        """
        Lista de alertas (mais recentes primeiro).

        Args:
            dados_sinais: cotações com as colunas sinal e confianca.
            inicio: primeira data considerada (None = todo o histórico).
        """
        partes = []
        for _, grupo in dados_sinais.sort_values(["ticker", "data"]).groupby("ticker", sort=False):
            g = grupo.reset_index(drop=True)
            anterior = g["sinal"].shift(1)
            mudou = g["sinal"].isin(["compra", "venda"]) & (g["sinal"] != anterior)
            confiante = g["confianca"].fillna(100) >= self.confianca_minima
            preco_final = g["close"].shift(-self.horizonte)
            alertas = g.loc[mudou & confiante, ["data", "ticker", "sinal", "close", "confianca"]].copy()
            alertas["preco_final"] = preco_final[mudou & confiante]
            partes.append(alertas)
        colunas = ["data", "ticker", "ativo", "tipo", "preco", "confianca", "preco_final",
                   "retorno_pct", "resultado"]
        if not partes:
            return pd.DataFrame(columns=colunas)

        alertas = pd.concat(partes, ignore_index=True).rename(columns={"sinal": "tipo", "close": "preco"})
        if inicio is not None:
            alertas = alertas[alertas["data"] >= pd.Timestamp(inicio)]
        alertas["ativo"] = alertas["ticker"].map(lambda t: ATIVOS[t].nome if t in ATIVOS else t)
        alertas["retorno_pct"] = (alertas["preco_final"] / alertas["preco"] - 1) * 100
        acertou = np.where(alertas["tipo"] == "compra", alertas["retorno_pct"] > 0, alertas["retorno_pct"] < 0)
        alertas["resultado"] = np.where(alertas["retorno_pct"].isna(), "Aguardando",
                                        np.where(acertou, "Acerto", "Erro"))
        return alertas.sort_values("data", ascending=False)[colunas].reset_index(drop=True)

    @staticmethod
    def resumir(historico: pd.DataFrame) -> ResumoAlertas:
        avaliados = historico[historico["resultado"] != "Aguardando"]
        acertos = int((avaliados["resultado"] == "Acerto").sum())
        # Retorno "a favor" do alerta: positivo quando o movimento seguiu o sinal.
        a_favor = np.where(avaliados["tipo"] == "compra", avaliados["retorno_pct"], -avaliados["retorno_pct"])
        por_ativo = (
            avaliados.assign(acerto=avaliados["resultado"] == "Acerto")
            .groupby("ticker")
            .agg(alertas=("acerto", "size"), acertos=("acerto", "sum"))
            .assign(taxa_acerto=lambda t: t["acertos"] / t["alertas"])
            .reset_index()
        )
        return ResumoAlertas(
            total=int(len(historico)),
            avaliados=int(len(avaliados)),
            acertos=acertos,
            aguardando=int((historico["resultado"] == "Aguardando").sum()),
            taxa_acerto=acertos / len(avaliados) if len(avaliados) else None,
            retorno_medio_pct=float(np.mean(a_favor)) if len(avaliados) else None,
            por_ativo=por_ativo,
        )
