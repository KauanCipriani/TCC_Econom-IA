"""
Cliente HTTP base com tempo-limite e cópia local das respostas.

Fluxo de uma consulta:
  1. se existe cópia local mais nova que a "validade", ela é usada (evita
     requisições repetidas a cada interação na interface);
  2. senão, a API é consultada e a resposta é gravada em disco;
  3. se a API falhar, usa-se a última cópia local, mesmo antiga;
  4. sem API e sem cópia, é lançada ErroIntegracao.

No modo offline (opção "Buscar dados na internet" desligada na interface), o
passo 2 é pulado e apenas as cópias locais são usadas.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from econom_ia.config import CAMINHOS


class ErroIntegracao(RuntimeError):
    """A fonte externa não respondeu e não há cópia local disponível."""


@dataclass
class Resposta:
    """Conteúdo obtido e de onde ele veio."""

    conteudo: Any
    origem: str            # "online", "cache" (cópia recente) ou "cache_antigo"
    obtido_em: datetime

    @property
    def atualizado(self) -> bool:
        return self.origem != "cache_antigo"


@dataclass
class SerieTemporal:
    """Série de valores datados (colunas data e valor) e sua procedência."""

    nome: str
    dados: pd.DataFrame
    origem: str
    obtido_em: datetime

    @property
    def ultimo_valor(self) -> float | None:
        return float(self.dados["valor"].iloc[-1]) if len(self.dados) else None

    @property
    def ultima_data(self) -> pd.Timestamp | None:
        return self.dados["data"].iloc[-1] if len(self.dados) else None

    def valor_em(self, dias_atras: int) -> float | None:
        """Último valor disponível até "dias_atras" dias antes da última data."""
        if not len(self.dados):
            return None
        limite = self.dados["data"].iloc[-1] - pd.Timedelta(days=dias_atras)
        anteriores = self.dados[self.dados["data"] <= limite]
        return float(anteriores["valor"].iloc[-1]) if len(anteriores) else None


class ClienteHTTP:
    """Base das integrações: requisição, validação e cache em disco."""

    nome: str = "http"
    TEMPO_LIMITE_S: float = 10.0
    # Parâmetros que mudam a cada dia (datas de início e fim) e por isso ficam fora
    # do nome da cópia local; assim a cópia de ontem continua valendo sem internet.
    IGNORAR_NA_CHAVE: tuple[str, ...] = ()
    CABECALHOS = {"User-Agent": "Mozilla/5.0 (Econom-IA; TCC Unoesc)"}

    def __init__(self, pasta_cache: Path | None = None, validade_s: int = 900,
                 sessao: requests.Session | None = None, offline: bool = False) -> None:
        """
        Args:
            pasta_cache: onde gravar as cópias locais das respostas.
            validade_s: por quantos segundos uma cópia é considerada atual.
            sessao: sessão HTTP (permite simular respostas nos testes).
            offline: se verdadeiro, nunca acessa a rede e usa só as cópias locais.
        """
        self.pasta_cache = Path(pasta_cache or CAMINHOS.cache_apis)
        self.validade_s = validade_s
        self.sessao = sessao or requests.Session()
        self.offline = offline

    # ── Requisições ────────────────────────────────────────────────────────
    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        resposta = self.sessao.get(url, params=params, headers=self.CABECALHOS, timeout=self.TEMPO_LIMITE_S)
        resposta.raise_for_status()
        return resposta

    def obter_json(self, url: str, params: dict | None = None) -> Resposta:
        return self.consultar(self._chave(url, params), lambda: self._get(url, params).json())

    def obter_texto(self, url: str, params: dict | None = None) -> Resposta:
        return self.consultar(self._chave(url, params), lambda: self._get(url, params).text)

    def consultar(self, chave: str, buscar: Callable[[], Any]) -> Resposta:
        """
        Executa "buscar" respeitando a política de cache descrita no módulo.

        O conteúdo devolvido por "buscar" precisa ser serializável em JSON.
        """
        copia = self._ler_cache(chave)
        if copia is not None and time.time() - copia["timestamp"] < self.validade_s:
            return Resposta(copia["conteudo"], "cache", datetime.fromtimestamp(copia["timestamp"]))
        if self.offline:
            if copia is not None:
                return Resposta(copia["conteudo"], "cache_antigo", datetime.fromtimestamp(copia["timestamp"]))
            raise ErroIntegracao(f"{self.nome}: modo offline e nenhuma cópia local disponível.")
        try:
            conteudo = buscar()
            self._gravar_cache(chave, conteudo)
            return Resposta(conteudo, "online", datetime.now())
        except Exception as erro:  # rede, formato inesperado, biblioteca ausente
            if copia is not None:
                return Resposta(copia["conteudo"], "cache_antigo", datetime.fromtimestamp(copia["timestamp"]))
            raise ErroIntegracao(f"{self.nome}: fonte indisponível ({erro.__class__.__name__}).") from erro

    # ── Cache em disco ─────────────────────────────────────────────────────
    def _chave(self, url: str, params: dict | None) -> str:
        estaveis = {k: v for k, v in (params or {}).items() if k not in self.IGNORAR_NA_CHAVE}
        texto = url + json.dumps(estaveis, sort_keys=True)
        return f"{self.nome}_{hashlib.sha1(texto.encode()).hexdigest()[:16]}"

    def _ler_cache(self, chave: str) -> dict | None:
        arquivo = self.pasta_cache / f"{chave}.json"
        if not arquivo.exists():
            return None
        try:
            return json.loads(arquivo.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _gravar_cache(self, chave: str, conteudo: Any) -> None:
        try:
            self.pasta_cache.mkdir(parents=True, exist_ok=True)
            (self.pasta_cache / f"{chave}.json").write_text(
                json.dumps({"timestamp": time.time(), "conteudo": conteudo}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass  # sem permissão de escrita: segue sem cache
