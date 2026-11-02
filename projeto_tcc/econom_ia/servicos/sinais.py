"""
Geração dos sinais de compra, venda e neutralidade exibidos na interface.

O serviço carrega os modelos treinados (arquivos .pkl) e o arquivo de
métricas, aplica o modelo escolhido às cotações e informa a confiança de
cada sinal. Também explica o sinal atual: para cada indicador, mede quanto a
probabilidade da classe prevista cai quando o indicador é trocado pelo seu
valor típico (mediana do histórico do ativo). O indicador com a maior queda
é o que "mais pesou" na decisão.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from econom_ia.config import CAMINHOS, CLASSES, EXPERIMENTO, Caminhos
from econom_ia.features import CalculadoraIndicadores, RegraLimiar
from econom_ia.modelos import ModeloClassificacao, ModeloLSTMClassificacao, carregar_modelo

NOMES_MODELOS = {
    "random_forest": "Random Forest",
    "svm": "SVM",
    "mlp": "MLP (PyTorch)",
    "lstm": "LSTM",
}


class ServicoSinais:
    """Acesso aos modelos treinados e geração de sinais."""

    def __init__(self, caminhos: Caminhos = CAMINHOS) -> None:
        self.caminhos = caminhos
        self.metricas: dict | None = self._ler_metricas()
        self._modelos: dict[str, ModeloClassificacao] = {}
        self.erros: dict[str, str] = {}

    # ── Metadados do treinamento ───────────────────────────────────────────
    def _ler_metricas(self) -> dict | None:
        try:
            return json.loads(self.caminhos.metricas.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @property
    def treinado(self) -> bool:
        return self.metricas is not None

    @property
    def principal(self) -> str:
        return (self.metricas or {}).get("modelo_principal", "random_forest")

    @property
    def horizonte(self) -> int:
        return (self.metricas or {}).get("info", {}).get("horizonte", EXPERIMENTO.horizonte)

    @property
    def regra_limiar(self) -> RegraLimiar:
        info = (self.metricas or {}).get("info", {})
        if "limiar" in info:
            return RegraLimiar.de_dict(info["limiar"])
        return RegraLimiar(*EXPERIMENTO.limiar_padrao)

    def disponiveis(self) -> dict[str, str]:
        """Modelos com arquivo salvo, na ordem de exibição."""
        return {chave: nome for chave, nome in NOMES_MODELOS.items()
                if (self.caminhos.modelos / f"{chave}.pkl").exists()}

    def modelo(self, chave: str | None = None) -> ModeloClassificacao | None:
        """Carrega (uma única vez) e devolve o modelo; None se não for possível."""
        chave = chave or self.principal
        if chave not in self._modelos and chave not in self.erros:
            try:
                self._modelos[chave] = carregar_modelo(self.caminhos.modelos / f"{chave}.pkl")
            except FileNotFoundError:
                self.erros[chave] = "Modelo ainda não treinado (execute python treinar_modelos.py)."
            except ImportError:
                self.erros[chave] = "Biblioteca PyTorch não instalada (pip install torch)."
            except Exception as erro:  # arquivo de versão incompatível, por exemplo
                self.erros[chave] = f"Não foi possível carregar o modelo ({erro.__class__.__name__})."
        return self._modelos.get(chave)

    # ── Sinais ─────────────────────────────────────────────────────────────
    def gerar(self, dados: pd.DataFrame, chave: str | None = None) -> pd.DataFrame:
        """
        Acrescenta a dados as colunas sinal, confianca (0 a 100) e p_<classe>.

        Linhas sem todos os indicadores (início do histórico) ficam sem sinal.
        A LSTM precisa do histórico anterior de cada ativo, por isso "dados"
        deve conter a série completa, e não apenas os dias de interesse.
        """
        saida = dados.copy()
        saida["sinal"] = None
        saida["confianca"] = np.nan
        for classe in CLASSES:
            saida[f"p_{classe}"] = np.nan
        modelo = self.modelo(chave)
        if modelo is None:
            return saida

        if isinstance(modelo, ModeloLSTMClassificacao):
            classes = modelo.prever(saida)
            validas = pd.notna(classes)
            saida.loc[validas, "sinal"] = classes[validas]
            for classe in CLASSES:
                saida.loc[validas, f"p_{classe}"] = (classes[validas] == classe).astype(float)
            return saida

        completas = saida[modelo.features].notna().all(axis=1)
        if completas.any():
            linhas = saida.loc[completas]
            probabilidades = modelo.prever_proba(linhas)
            previsto = modelo.prever(linhas)
            saida.loc[completas, "sinal"] = previsto
            for classe in CLASSES:
                saida.loc[completas, f"p_{classe}"] = probabilidades[classe].to_numpy()
            confianca = probabilidades.to_numpy()[np.arange(len(previsto)),
                                                   [list(CLASSES).index(c) for c in previsto]]
            saida.loc[completas, "confianca"] = np.round(confianca * 100, 1)
        return saida

    def explicar(self, dados_ativo: pd.DataFrame, chave: str | None = None) -> list[dict]:
        """
        Contribuição de cada indicador para o sinal do último pregão.

        Returns:
            Lista ordenada (maior contribuição primeiro) de dicionários com
            feature, descricao, valor e contribuicao (em pontos percentuais).
        """
        modelo = self.modelo(chave)
        if modelo is None or isinstance(modelo, ModeloLSTMClassificacao):
            return []
        historico = dados_ativo.dropna(subset=modelo.features)
        if historico.empty:
            return []
        ultima = historico.iloc[[-1]]
        classe = modelo.prever(ultima)[0]
        base = float(modelo.prever_proba(ultima)[classe].iloc[0])

        referencia = historico[modelo.features].median()
        alteradas = pd.concat([ultima] * len(modelo.features), ignore_index=True)
        for i, feature in enumerate(modelo.features):
            alteradas.loc[i, feature] = referencia[feature]
        sem_feature = modelo.prever_proba(alteradas)[classe].to_numpy()

        contribuicoes = [
            {
                "feature": feature,
                "descricao": CalculadoraIndicadores.DESCRICOES.get(feature, feature),
                "valor": float(ultima[feature].iloc[0]),
                "contribuicao": round((base - sem_feature[i]) * 100, 2),
            }
            for i, feature in enumerate(modelo.features)
        ]
        return sorted(contribuicoes, key=lambda c: -c["contribuicao"])

    def limiar_atual(self, linha: pd.Series) -> float:
        """Limiar (em fração) que define "movimento relevante" para a linha."""
        regra = self.regra_limiar
        if regra.tipo == "fixo":
            return regra.valor
        return regra.valor * float(linha["volatilidade_20"]) * np.sqrt(self.horizonte)
