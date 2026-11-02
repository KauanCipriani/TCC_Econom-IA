"""
Configurações centrais do sistema.

Tudo o que é parâmetro (ativos monitorados, período do estudo, horizonte e
limiar da variável-alvo, caminhos de arquivos) fica concentrado aqui, para que
o restante do código não tenha "números mágicos" espalhados.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Pasta "projeto_tcc" (um nível acima deste pacote). Os caminhos são relativos
# a ela, o que permite rodar o sistema a partir de qualquer diretório.
RAIZ_PROJETO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Ativo:
    """Cadastro de uma ação monitorada pelo sistema."""

    ticker: str          # código na B3 (ex.: ITUB4)
    nome: str            # nome da instituição exibido na interface
    cor: str             # cor usada nos gráficos
    simbolo_yahoo: str   # símbolo no Yahoo Finance (sufixo .SA)


# Ordem de exibição na interface. A SANB11 é a ação reservada para o teste
# out-of-sample: ela nunca participa do treinamento dos modelos.
# As cores seguem uma paleta categórica validada para daltonismo no tema escuro.
ATIVOS: dict[str, Ativo] = {
    "ITUB4": Ativo("ITUB4", "Itaú Unibanco", "#3987e5", "ITUB4.SA"),
    "BBDC4": Ativo("BBDC4", "Bradesco", "#d95926", "BBDC4.SA"),
    "BBAS3": Ativo("BBAS3", "Banco do Brasil", "#199e70", "BBAS3.SA"),
    "SANB11": Ativo("SANB11", "Santander Brasil", "#c98500", "SANB11.SA"),
}


@dataclass(frozen=True)
class Caminhos:
    """Localização dos arquivos persistidos (arquitetura baseada em arquivos)."""

    raiz: Path = RAIZ_PROJETO

    @property
    def dados(self) -> Path:
        return self.raiz / "dados"

    @property
    def modelos(self) -> Path:
        return self.raiz / "modelos"

    @property
    def cache_apis(self) -> Path:
        """Cópias locais das últimas respostas das APIs (uso offline)."""
        return self.dados / "cache_apis"

    @property
    def ohlcv(self) -> Path:
        return self.dados / "b3_financeiro_ohlcv.parquet"

    @property
    def features(self) -> Path:
        return self.dados / "b3_financeiro_features.parquet"

    @property
    def metricas(self) -> Path:
        return self.modelos / "metricas.json"

    @property
    def previsoes_avaliacao(self) -> Path:
        """Previsões dos modelos nos conjuntos de avaliação (tela de desempenho)."""
        return self.modelos / "previsoes_avaliacao.parquet"


@dataclass(frozen=True)
class ConfigExperimento:
    """Parâmetros do experimento descritos na metodologia do artigo."""

    # Período do estudo (jan/2019 a set/2025) e período de avaliação
    # prospectiva, formado pelos pregões posteriores ao fim do estudo.
    inicio_estudo: str = "2019-01-01"
    fim_estudo: str = "2025-09-30"

    tickers_treino: tuple[str, ...] = ("ITUB4", "BBDC4", "BBAS3")
    ticker_oos: str = "SANB11"

    # Variável-alvo: retorno em "horizonte" pregões comparado a +/- limiar.
    # O TC I usou limiar fixo de 1,5%; no TC II são avaliados outros limiares
    # fixos e um limiar adaptativo à volatilidade (tipo, valor).
    horizonte: int = 5
    limiar_padrao: tuple[str, float] = ("fixo", 0.015)
    limiares_avaliados: tuple[tuple[str, float], ...] = (
        ("fixo", 0.010), ("fixo", 0.015), ("fixo", 0.020),
        ("fixo", 0.025), ("fixo", 0.030), ("volatilidade", 0.5),
    )

    # Divisão temporal e validação cruzada.
    proporcao_treino: float = 0.70
    n_folds: int = 5

    # Previsão de preço (LSTM): horizontes de 1 a 15 pregões.
    horizonte_maximo_previsao: int = 15

    semente: int = 42
    n_jobs: int = 2


# Classes da variável-alvo, sempre nesta ordem (usada nas matrizes de confusão).
CLASSES: tuple[str, str, str] = ("venda", "neutro", "compra")

# Aviso regulatório exibido na interface.
AVISO_CVM = (
    "Ferramenta acadêmica de apoio à decisão. Não constitui recomendação de "
    "investimento. Consulte um profissional credenciado pela CVM antes de "
    "tomar decisões financeiras."
)

CAMINHOS = Caminhos()
EXPERIMENTO = ConfigExperimento()
