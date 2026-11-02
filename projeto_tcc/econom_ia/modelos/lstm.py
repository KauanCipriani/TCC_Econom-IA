"""
Rede LSTM (Long Short-Term Memory) para a predição de preço.

A rede recebe uma janela com os últimos "janela" pregões de um ativo
(retornos e indicadores padronizados) e estima, de uma só vez, o retorno
logarítmico acumulado para cada horizonte de 1 a H pregões. O preço previsto
para o horizonte h é obtido por P(t+h) = P(t) * exp(retorno previsto).
Estimar retornos, e não o preço em reais, torna o problema independente do
nível de preço de cada ação e evita que a rede apenas "copie" o último preço.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from econom_ia.config import CLASSES
from econom_ia.modelos.base import ModeloClassificacao


def _criar_rede(n_entradas: int, oculto: int, camadas: int, dropout: float, horizontes: int):
    from torch import nn

    class RedeLSTM(nn.Module):
        """LSTM seguida de uma camada linear com uma saída por horizonte."""

        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                n_entradas, oculto, num_layers=camadas, batch_first=True,
                dropout=dropout if camadas > 1 else 0.0,
            )
            self.saida = nn.Sequential(nn.Dropout(dropout), nn.Linear(oculto, horizontes))

        def forward(self, x):
            sequencia, _ = self.lstm(x)
            return self.saida(sequencia[:, -1, :])

    return RedeLSTM()


class PrevisorLSTM:
    """Treina a LSTM e gera previsões de retorno (e de preço) por horizonte."""

    FEATURES_SEQUENCIA: list[str] = [
        "ret_1", "ret_5", "rsi_14", "macd_hist_z", "bb_position",
        "sma_21_ratio", "vol_rel_20", "volume_z20",
    ]

    def __init__(
        self,
        janela: int = 30,
        horizontes: int = 15,
        oculto: int = 64,
        camadas: int = 1,
        dropout: float = 0.2,
        taxa_aprendizado: float = 1e-3,
        epocas: int = 80,
        tamanho_lote: int = 128,
        paciencia: int = 10,
        fracao_validacao: float = 0.15,
        semente: int = 42,
    ) -> None:
        self.janela = janela
        self.horizontes = horizontes
        self.oculto = oculto
        self.camadas = camadas
        self.dropout = dropout
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        self.paciencia = paciencia
        self.fracao_validacao = fracao_validacao
        self.semente = semente
        # Estado aprendido no treino.
        self.media_x = None
        self.desvio_x = None
        self.media_y = None
        self.desvio_y = None
        self.rede = None
        self.perda_validacao: float | None = None
        self.epocas_treinadas: int = 0

    # ── Montagem das sequências ────────────────────────────────────────────
    def _janelas(self, df: pd.DataFrame, com_alvo: bool):
        """
        Constrói as janelas deslizantes por ativo.

        Returns:
            X (n, janela, variáveis), Y (n, horizontes) ou None e uma tabela
            com ticker, data e preço de fechamento de cada janela.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        lista_x, lista_y, chaves = [], [], []
        for ticker, grupo in df.sort_values(["ticker", "data"]).groupby("ticker", sort=False):
            grupo = grupo.dropna(subset=self.FEATURES_SEQUENCIA)
            n = len(grupo)
            if n < self.janela + (self.horizontes if com_alvo else 0):
                continue
            x = (grupo[self.FEATURES_SEQUENCIA].to_numpy(dtype=np.float32) - self.media_x) / self.desvio_x
            janelas = sliding_window_view(x, self.janela, axis=0).transpose(0, 2, 1)
            log_preco = np.log(grupo["close"].to_numpy(dtype=np.float64))
            finais = np.arange(self.janela - 1, n)  # índice do último pregão de cada janela
            if com_alvo:
                finais = finais[finais + self.horizontes < n]
                janelas = janelas[: len(finais)]
                h = np.arange(1, self.horizontes + 1)
                lista_y.append((log_preco[finais[:, None] + h] - log_preco[finais, None]).astype(np.float32))
            lista_x.append(janelas.astype(np.float32))
            chaves.append(pd.DataFrame({
                "ticker": ticker,
                "data": grupo["data"].to_numpy()[finais],
                "close": grupo["close"].to_numpy()[finais],
            }))
        if not lista_x:
            return None, None, pd.DataFrame(columns=["ticker", "data", "close"])
        X = np.concatenate(lista_x)
        Y = np.concatenate(lista_y) if com_alvo else None
        return X, Y, pd.concat(chaves, ignore_index=True)

    # ── Treinamento ────────────────────────────────────────────────────────
    def ajustar(self, df: pd.DataFrame) -> PrevisorLSTM:
        import torch

        torch.manual_seed(self.semente)
        base = df.dropna(subset=self.FEATURES_SEQUENCIA)
        self.media_x = base[self.FEATURES_SEQUENCIA].mean().to_numpy(dtype=np.float32)
        self.desvio_x = base[self.FEATURES_SEQUENCIA].std().replace(0, 1).to_numpy(dtype=np.float32)

        X, Y, chaves = self._janelas(df, com_alvo=True)
        if X is None:
            raise ValueError("Dados insuficientes para treinar a LSTM.")

        # Validação: janelas das datas mais recentes, com expurgo de
        # "horizontes" pregões para que os alvos do treino não as alcancem.
        datas = np.sort(chaves["data"].unique())
        corte = int(len(datas) * (1 - self.fracao_validacao))
        inicio_val = datas[corte]
        limite_treino = datas[max(corte - self.horizontes, 0)]
        em_treino = (chaves["data"] < limite_treino).to_numpy()
        em_val = (chaves["data"] >= inicio_val).to_numpy()

        self.media_y = Y[em_treino].mean(axis=0)
        self.desvio_y = Y[em_treino].std(axis=0) + 1e-8
        Y_pad = (Y - self.media_y) / self.desvio_y

        x_treino, y_treino = torch.tensor(X[em_treino]), torch.tensor(Y_pad[em_treino])
        x_val, y_val = torch.tensor(X[em_val]), torch.tensor(Y_pad[em_val])

        rede = _criar_rede(len(self.FEATURES_SEQUENCIA), self.oculto, self.camadas, self.dropout, self.horizontes)
        otimizador = torch.optim.Adam(rede.parameters(), lr=self.taxa_aprendizado)
        perda_fn = torch.nn.MSELoss()
        gerador = torch.Generator().manual_seed(self.semente)

        melhor, melhor_estado, sem_melhora, executadas = np.inf, None, 0, 0
        for epoca in range(1, self.epocas + 1):
            executadas = epoca
            rede.train()
            ordem = torch.randperm(len(x_treino), generator=gerador)
            for inicio in range(0, len(ordem), self.tamanho_lote):
                lote = ordem[inicio:inicio + self.tamanho_lote]
                otimizador.zero_grad()
                perda = perda_fn(rede(x_treino[lote]), y_treino[lote])
                perda.backward()
                torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
                otimizador.step()
            rede.eval()
            with torch.no_grad():
                perda_val = float(perda_fn(rede(x_val), y_val))
            if perda_val < melhor - 1e-4:
                melhor, sem_melhora = perda_val, 0
                melhor_estado = {k: v.detach().clone() for k, v in rede.state_dict().items()}
            else:
                sem_melhora += 1
                if sem_melhora >= self.paciencia:
                    break

        if melhor_estado is not None:
            rede.load_state_dict(melhor_estado)
        rede.eval()
        self.rede = rede
        self.perda_validacao = float(melhor)
        self.epocas_treinadas = executadas
        return self

    # ── Inferência ─────────────────────────────────────────────────────────
    def prever(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorno logarítmico acumulado previsto para cada janela completa.

        Returns:
            DataFrame com ticker, data, close e colunas r_1 ... r_H.
        """
        import torch

        if self.rede is None:
            raise RuntimeError("A LSTM ainda não foi treinada.")
        X, _, chaves = self._janelas(df, com_alvo=False)
        colunas = [f"r_{h}" for h in range(1, self.horizontes + 1)]
        if X is None:
            return pd.DataFrame(columns=["ticker", "data", "close", *colunas])
        with torch.no_grad():
            saida = self.rede(torch.tensor(X)).numpy()
        previsto = saida * self.desvio_y + self.media_y
        return pd.concat([chaves, pd.DataFrame(previsto, columns=colunas)], axis=1)

    def prever_precos(self, df_ativo: pd.DataFrame, dias: int) -> np.ndarray:
        """Preços previstos para os próximos "dias" pregões a partir da última data."""
        previsao = self.prever(df_ativo)
        if previsao.empty:
            return np.array([])
        ultima = previsao.iloc[-1]
        retornos = np.array([ultima[f"r_{h}"] for h in range(1, dias + 1)], dtype=float)
        return float(ultima["close"]) * np.exp(retornos)

    # ── Persistência ───────────────────────────────────────────────────────
    def _configuracao(self) -> dict:
        return {
            "janela": self.janela, "horizontes": self.horizontes, "oculto": self.oculto,
            "camadas": self.camadas, "dropout": self.dropout,
            "taxa_aprendizado": self.taxa_aprendizado, "epocas": self.epocas,
            "tamanho_lote": self.tamanho_lote, "paciencia": self.paciencia,
            "fracao_validacao": self.fracao_validacao, "semente": self.semente,
        }

    def __getstate__(self):
        estado = self.__dict__.copy()
        rede = estado.pop("rede", None)
        if rede is not None:
            import torch

            buffer = io.BytesIO()
            torch.save(rede.state_dict(), buffer)
            estado["_pesos_rede"] = buffer.getvalue()
        return estado

    def __setstate__(self, estado):
        pesos = estado.pop("_pesos_rede", None)
        self.__dict__.update(estado)
        self.rede = None
        if pesos is not None:
            import torch

            rede = _criar_rede(len(self.FEATURES_SEQUENCIA), self.oculto, self.camadas, self.dropout, self.horizontes)
            rede.load_state_dict(torch.load(io.BytesIO(pesos), weights_only=True))
            rede.eval()
            self.rede = rede

    def descricao(self) -> dict:
        return {**self._configuracao(), "epocas_treinadas": self.epocas_treinadas,
                "perda_validacao": self.perda_validacao}


class ModeloLSTMClassificacao(ModeloClassificacao):
    """
    Usa a previsão de retorno da LSTM para emitir as mesmas três classes.

    Como a rede tende a prever retornos próximos de zero, os pontos de corte
    entre as classes são calibrados no treino para reproduzir a proporção de
    compra, neutro e venda observada nos rótulos, tornando a comparação com
    os classificadores diretos mais justa.
    """

    chave = "lstm"
    nome = "LSTM"

    def __init__(self, previsor: PrevisorLSTM, horizonte: int = 5, semente: int = 42) -> None:
        super().__init__(features=PrevisorLSTM.FEATURES_SEQUENCIA, semente=semente)
        self.previsor = previsor
        self.horizonte = horizonte
        self.corte_venda: float | None = None
        self.corte_compra: float | None = None

    def _construir(self, para_busca: bool = False):  # não usa estimador do scikit-learn
        return None

    def grade_busca(self) -> dict:
        return {}

    def ajustar(self, df: pd.DataFrame) -> ModeloLSTMClassificacao:
        import time

        inicio = time.perf_counter()
        self.previsor.ajustar(df)
        self.tempo_treino_s = time.perf_counter() - inicio
        retornos = self._retorno_previsto(df)
        rotulados = df.assign(_r=retornos).dropna(subset=["alvo", "_r"])
        proporcao = rotulados["alvo"].value_counts(normalize=True)
        self.corte_venda = float(np.quantile(rotulados["_r"], proporcao.get("venda", 0.0)))
        self.corte_compra = float(np.quantile(rotulados["_r"], 1 - proporcao.get("compra", 0.0)))
        self.estimador = self.previsor
        self.hiperparametros = self.previsor.descricao()
        return self

    def _retorno_previsto(self, df: pd.DataFrame) -> np.ndarray:
        """Retorno previsto no horizonte da variável-alvo, alinhado às linhas de df."""
        previsao = self.previsor.prever(df)[["ticker", "data", f"r_{self.horizonte}"]]
        alinhado = df[["ticker", "data"]].merge(previsao, on=["ticker", "data"], how="left")
        return np.expm1(alinhado[f"r_{self.horizonte}"].to_numpy(dtype=float))

    def prever(self, df: pd.DataFrame) -> np.ndarray:
        self._verificar_treino()
        retornos = self._retorno_previsto(df)
        classes = np.where(retornos >= self.corte_compra, "compra",
                           np.where(retornos <= self.corte_venda, "venda", "neutro")).astype(object)
        classes[np.isnan(retornos)] = None
        return classes

    def prever_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """A LSTM não estima probabilidades de classe; devolve a classe com peso 1."""
        classes = self.prever(df)
        return pd.DataFrame({c: (classes == c).astype(float) for c in CLASSES}, index=df.index)

    def descricao_hiperparametros(self) -> dict:
        return {**self.previsor.descricao(), "corte_venda": self.corte_venda, "corte_compra": self.corte_compra}

    def caminho_arquivo(self, pasta: Path) -> Path:
        return Path(pasta) / "lstm.pkl"
