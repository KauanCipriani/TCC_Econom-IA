"""
Criação da variável-alvo de classificação.

Para cada pregão t calcula-se o retorno do preço de fechamento entre t e
t + horizonte. Se ele superar +limiar a classe é "compra"; se ficar abaixo de
-limiar a classe é "venda"; caso contrário, "neutro". Os últimos "horizonte"
pregões de cada ativo ainda não têm preço futuro conhecido e ficam sem rótulo.

O limiar pode ser:
  * fixo: o mesmo percentual para todos os dias (ex.: 1,5% no TC I);
  * adaptativo: proporcional à volatilidade recente do ativo, isto é,
    fator x desvio-padrão dos retornos diários de 20 pregões x raiz(horizonte).
    Assim o que se considera um "movimento relevante" acompanha o regime de
    volatilidade do mercado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from econom_ia.config import CLASSES


@dataclass(frozen=True)
class RegraLimiar:
    """Define como o limiar entre as classes é calculado."""

    tipo: str = "fixo"      # "fixo" ou "volatilidade"
    valor: float = 0.015    # percentual (fixo) ou fator multiplicativo (volatilidade)

    def __post_init__(self) -> None:
        if self.tipo not in ("fixo", "volatilidade"):
            raise ValueError("O tipo de limiar deve ser 'fixo' ou 'volatilidade'.")
        if self.valor <= 0:
            raise ValueError("O valor do limiar deve ser positivo.")

    def descricao(self) -> str:
        if self.tipo == "fixo":
            return f"±{self.valor * 100:.1f}%".replace(".", ",")
        return f"{self.valor:.2f} x volatilidade".replace(".", ",")

    def como_dict(self) -> dict:
        return {"tipo": self.tipo, "valor": self.valor, "descricao": self.descricao()}

    @classmethod
    def de_dict(cls, dados: dict) -> RegraLimiar:
        return cls(tipo=dados["tipo"], valor=float(dados["valor"]))


class Rotulador:
    """Gera a variável-alvo por regra de limiar sobre o retorno futuro."""

    def __init__(self, horizonte: int = 5, regra: RegraLimiar | None = None) -> None:
        if horizonte < 1:
            raise ValueError("O horizonte deve ser de pelo menos 1 pregão.")
        self.horizonte = horizonte
        self.regra = regra or RegraLimiar()

    def limiares(self, df: pd.DataFrame) -> np.ndarray:
        """Limiar (em fração) aplicável a cada linha da base."""
        if self.regra.tipo == "fixo":
            return np.full(len(df), self.regra.valor)
        if "volatilidade_20" not in df.columns:
            raise ValueError("O limiar adaptativo requer a coluna 'volatilidade_20'.")
        return self.regra.valor * df["volatilidade_20"].to_numpy(dtype=float) * np.sqrt(self.horizonte)

    def adicionar_alvo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Acrescenta 'retorno_futuro', 'limiar_alvo' e 'alvo' (calculados por ativo)."""
        saida = df.sort_values(["ticker", "data"]).reset_index(drop=True).copy()
        futuro = saida.groupby("ticker", sort=False)["close"].shift(-self.horizonte)
        saida["retorno_futuro"] = futuro / saida["close"] - 1
        saida["limiar_alvo"] = self.limiares(saida)
        saida["alvo"] = self.classificar(saida["retorno_futuro"].to_numpy(), saida["limiar_alvo"].to_numpy())
        return saida

    def classificar(self, retornos, limiares) -> np.ndarray:
        """Converte retornos em classes; retornos ou limiares ausentes geram rótulo ausente."""
        retornos = np.asarray(retornos, dtype=float)
        limiares = np.broadcast_to(np.asarray(limiares, dtype=float), retornos.shape)
        rotulos = np.where(
            retornos > limiares, "compra",
            np.where(retornos < -limiares, "venda", "neutro"),
        ).astype(object)
        rotulos[np.isnan(retornos) | np.isnan(limiares)] = None
        return rotulos

    @staticmethod
    def distribuicao(alvo: pd.Series) -> dict[str, float]:
        """Proporção de cada classe (ignora rótulos ausentes)."""
        contagem = pd.Series(alvo).dropna().value_counts(normalize=True)
        return {classe: round(float(contagem.get(classe, 0.0)), 4) for classe in CLASSES}
