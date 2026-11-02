"""Random Forest: comitê de árvores de decisão que votam na classe final."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from econom_ia.modelos.base import ModeloClassificacao


class ModeloRandomForest(ModeloClassificacao):
    chave = "random_forest"
    nome = "Random Forest"

    def _construir(self, para_busca: bool = False):
        return RandomForestClassifier(
            n_estimators=300,
            # Pesos por classe recalculados em cada amostra bootstrap, para
            # compensar o desbalanceamento natural entre as três classes.
            class_weight="balanced_subsample",
            random_state=self.semente,
            # Na busca em grade o paralelismo fica com o GridSearchCV.
            n_jobs=1 if para_busca else self.n_jobs,
        )

    def grade_busca(self) -> dict:
        return {
            "max_depth": [3, 5, 8, 12],
            "min_samples_leaf": [10, 30, 80],
            "max_features": ["sqrt", 0.5],
        }

    def importancias(self) -> dict[str, float] | None:
        if self.estimador is None:
            return None
        pares = zip(self.features, self.estimador.feature_importances_, strict=True)
        return {nome: float(valor) for nome, valor in sorted(pares, key=lambda p: -p[1])}


class ModeloRandomForestTC1(ModeloRandomForest):
    """
    Configuração exata dos resultados parciais (TC I), mantida apenas para
    servir de referência na comparação com os modelos ajustados no TC II.
    """

    chave = "random_forest_tc1"
    nome = "Random Forest (configuração do TC I)"

    def _construir(self, para_busca: bool = False):
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=self.semente,
            n_jobs=self.n_jobs,
        )

    def grade_busca(self) -> dict:
        return {}
