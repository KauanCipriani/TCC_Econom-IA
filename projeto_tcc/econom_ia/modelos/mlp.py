"""
MLP (Perceptron Multicamadas) implementado em PyTorch.

A rede é embrulhada em um estimador compatível com o scikit-learn
(ClassificadorMLPTorch), o que permite usá-la no mesmo pipeline de
padronização, busca de hiperparâmetros e validação dos demais modelos.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from econom_ia.modelos.base import ModeloClassificacao


def _construir_rede(n_entradas: int, camadas: tuple[int, ...], dropout: float, n_saidas: int):
    from torch import nn

    blocos, anterior = [], n_entradas
    for neuronios in camadas:
        blocos += [nn.Linear(anterior, neuronios), nn.ReLU(), nn.Dropout(dropout)]
        anterior = neuronios
    blocos.append(nn.Linear(anterior, n_saidas))
    return nn.Sequential(*blocos)


class ClassificadorMLPTorch(ClassifierMixin, BaseEstimator):
    """
    Classificador MLP com parada antecipada (early stopping).

    Os últimos "fracao_validacao" das amostras (as mais recentes, pois os dados
    chegam em ordem cronológica) são separados para monitorar a perda de
    validação; o treino para quando ela deixa de melhorar por "paciencia"
    épocas e os pesos da melhor época são restaurados.
    """

    def __init__(
        self,
        camadas: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        taxa_aprendizado: float = 1e-3,
        decaimento_peso: float = 1e-4,
        epocas: int = 200,
        tamanho_lote: int = 256,
        paciencia: int = 15,
        fracao_validacao: float = 0.15,
        semente: int = 42,
    ) -> None:
        self.camadas = camadas
        self.dropout = dropout
        self.taxa_aprendizado = taxa_aprendizado
        self.decaimento_peso = decaimento_peso
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        self.paciencia = paciencia
        self.fracao_validacao = fracao_validacao
        self.semente = semente

    def fit(self, X, y):  # noqa: N803 (convenção do scikit-learn)
        import torch

        torch.manual_seed(self.semente)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=object)
        self.classes_ = np.unique(y)
        y_indice = np.searchsorted(self.classes_, y)
        self.n_features_in_ = X.shape[1]

        corte = int(len(X) * (1 - self.fracao_validacao))
        X_treino, y_treino = torch.tensor(X[:corte]), torch.tensor(y_indice[:corte])
        X_val, y_val = torch.tensor(X[corte:]), torch.tensor(y_indice[corte:])

        # Pesos inversamente proporcionais à frequência de cada classe.
        contagem = np.bincount(y_indice[:corte], minlength=len(self.classes_)).astype(np.float32)
        pesos = torch.tensor(contagem.sum() / (len(self.classes_) * np.maximum(contagem, 1)))
        perda_fn = torch.nn.CrossEntropyLoss(weight=pesos)

        rede = _construir_rede(X.shape[1], tuple(self.camadas), self.dropout, len(self.classes_))
        otimizador = torch.optim.Adam(
            rede.parameters(), lr=self.taxa_aprendizado, weight_decay=self.decaimento_peso)
        gerador = torch.Generator().manual_seed(self.semente)

        melhor_perda, melhor_estado, sem_melhora = np.inf, None, 0
        for _ in range(self.epocas):
            rede.train()
            ordem = torch.randperm(len(X_treino), generator=gerador)
            for inicio in range(0, len(ordem), self.tamanho_lote):
                lote = ordem[inicio:inicio + self.tamanho_lote]
                otimizador.zero_grad()
                perda = perda_fn(rede(X_treino[lote]), y_treino[lote])
                perda.backward()
                otimizador.step()
            rede.eval()
            with torch.no_grad():
                perda_val = float(perda_fn(rede(X_val), y_val)) if len(X_val) else float(perda)
            if perda_val < melhor_perda - 1e-4:
                melhor_perda, sem_melhora = perda_val, 0
                melhor_estado = {k: v.detach().clone() for k, v in rede.state_dict().items()}
            else:
                sem_melhora += 1
                if sem_melhora >= self.paciencia:
                    break

        if melhor_estado is not None:
            rede.load_state_dict(melhor_estado)
        rede.eval()
        self.rede_ = rede
        return self

    def predict_proba(self, X):  # noqa: N803
        import torch

        with torch.no_grad():
            logits = self.rede_(torch.tensor(np.asarray(X, dtype=np.float32)))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):  # noqa: N803
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # A rede é guardada como arrays NumPy ao serializar com joblib, o que
    # torna o arquivo independente dos detalhes internos do PyTorch.
    def __getstate__(self):
        estado = self.__dict__.copy()
        rede = estado.pop("rede_", None)
        if rede is not None:
            estado["_pesos_rede"] = {k: v.cpu().numpy() for k, v in rede.state_dict().items()}
        return estado

    def __setstate__(self, estado):
        pesos = estado.pop("_pesos_rede", None)
        self.__dict__.update(estado)
        if pesos is not None:
            import torch

            rede = _construir_rede(
                self.n_features_in_, tuple(self.camadas), self.dropout, len(self.classes_))
            rede.load_state_dict({k: torch.tensor(v) for k, v in pesos.items()})
            rede.eval()
            self.rede_ = rede


class ModeloMLP(ModeloClassificacao):
    chave = "mlp"
    nome = "MLP (PyTorch)"

    def _construir(self, para_busca: bool = False):
        return Pipeline([
            ("padronizacao", StandardScaler()),
            ("modelo", ClassificadorMLPTorch(semente=self.semente)),
        ])

    def grade_busca(self) -> dict:
        return {
            "modelo__camadas": [(32,), (64, 32)],
            "modelo__dropout": [0.1, 0.3],
            "modelo__decaimento_peso": [1e-4, 1e-3],
        }
