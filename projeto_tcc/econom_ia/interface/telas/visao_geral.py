"""
Tela Visão Geral: primeiro contato do usuário com o sistema.

Abre com a resposta principal (como o setor bancário fechou e o que o modelo
indica para cada banco) e, abaixo, os indicadores de mercado (Ibovespa, dólar,
Selic e IPCA), o desempenho recente dos quatro bancos e as manchetes da semana.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from econom_ia.config import ATIVOS
from econom_ia.integracoes import SerieTemporal
from econom_ia.interface import dados_app
from econom_ia.interface import formatacao as fmt
from econom_ia.interface.telas.base import Tela


class TelaVisaoGeral(Tela):
    chave = "visao_geral"
    rotulo = "Visão Geral"
    icone = ":material/home:"
    SECOES_OPCIONAIS = {
        "sinais": "Sinal de hoje para cada banco",
        "mercado": "Mercado e juros",
        "grafico": "Gráfico de desempenho dos bancos",
        "manchetes": "Manchetes da semana",
    }

    # Variação média do setor (em %) acima da qual o dia é considerado de alta.
    LIMITE_HUMOR = 0.5

    def renderizar(self) -> None:
        dados = self.ctx.dados_com_sinais()
        ultimos = self._ultimos_por_ativo(dados)
        _, botao = st.columns([5, 1.25])
        with botao:
            self.personalizar()
        self._boas_vindas(ultimos)
        if self.mostrar("sinais"):
            self._sinais_de_hoje(ultimos)
        if self.mostrar("mercado"):
            self.ui.renderizar(self.ui.titulo_secao("Mercado e juros",
                                                    "referências para entender o momento dos bancos"))
            self._indicadores_mercado()
        if self.mostrar("grafico"):
            self.ui.renderizar(self.ui.titulo_secao(
                "Desempenho dos bancos nos últimos 6 meses",
                "cada linha começa em 100: acima disso, a ação subiu no período; abaixo, caiu"))
            self.exibir(self.graficos.desempenho_setor(dados, pregoes=126), "grafico_setor")
        if self.mostrar("manchetes"):
            self._manchetes()
        self.ui.renderizar(self.ui.aviso_cvm())

    # ── Dados auxiliares ───────────────────────────────────────────────────
    @staticmethod
    def _ultimos_por_ativo(dados: pd.DataFrame) -> list[dict]:
        """Último pregão de cada ativo, com variação no dia e sinal."""
        resultado = []
        for ticker, ativo in ATIVOS.items():
            serie = dados[dados["ticker"] == ticker].sort_values("data")
            if len(serie) < 2:
                continue
            ultimo, anterior = serie.iloc[-1], serie.iloc[-2]
            resultado.append({
                "ticker": ticker,
                "nome": ativo.nome,
                "data": ultimo["data"],
                "preco": float(ultimo["close"]),
                "variacao": (ultimo["close"] / anterior["close"] - 1) * 100,
                "sinal": ultimo["sinal"],
                "confianca": ultimo["confianca"],
            })
        return resultado

    # ── Seções ─────────────────────────────────────────────────────────────
    def _boas_vindas(self, ultimos: list[dict]) -> None:
        agora = datetime.now()
        saudacao = "Bom dia" if agora.hour < 12 else "Boa tarde" if agora.hour < 18 else "Boa noite"
        if not ultimos:
            return
        media = sum(item["variacao"] for item in ultimos) / len(ultimos)
        if media > self.LIMITE_HUMOR:
            humor = '<span class="sobe">▲ em alta</span>'
        elif media < -self.LIMITE_HUMOR:
            humor = '<span class="desce">▼ em baixa</span>'
        else:
            humor = '<span class="estavel">● estável</span>'
        origem = "ao vivo (Yahoo Finance)" if self.ctx.cotacoes.origem == "online" else "da base local"
        self.ui.renderizar(
            '<div class="boas-vindas">'
            f'<div class="boas-vindas-data">{fmt.data_extenso(agora)}</div>'
            f'<div class="boas-vindas-titulo">{saudacao}! O setor bancário está {humor}.</div>'
            f'<div class="boas-vindas-detalhe">Variação média dos quatro bancos no pregão de '
            f'{fmt.data_curta(ultimos[0]["data"])}: {self.ui.variacao(media)}. {self._resumo_sinais(ultimos)} '
            f'Cotações {origem}.</div>'
            '</div>'
        )

    def _resumo_sinais(self, ultimos: list[dict]) -> str:
        """Frase com a contagem dos sinais do dia (ex.: "venda para 3 dos 4 bancos")."""
        contagem = {"compra": 0, "neutro": 0, "venda": 0}
        for item in ultimos:
            contagem[item["sinal"] if item["sinal"] in contagem else "neutro"] += 1
        partes = []
        for sinal, texto in (("compra", "compra"), ("venda", "venda"), ("neutro", "neutro")):
            if contagem[sinal]:
                partes.append(f"<strong>{texto}</strong> para {contagem[sinal]}")
        juncao = partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + " e " + partes[-1]
        modelo = self.ctx.nome_modelo(self.ctx.modelo_principal)
        return f"O modelo {modelo} indica {juncao} dos {len(ultimos)} bancos."

    def _indicadores_mercado(self) -> None:
        """Quatro cartões: Ibovespa, dólar, Selic e IPCA (fontes externas)."""
        internet = self.ctx.usar_internet
        self.ui.renderizar(self.ui.grade([
            self._cartao_cotacao("Ibovespa", "ibovespa", casas=0, unidade="pts"),
            self._cartao_dolar(),
            self._cartao_selic(dados_app.serie_bcb("selic_meta", internet)),
            self._cartao_ipca(dados_app.serie_bcb("ipca_12m", internet)),
        ]))

    def _cartao_cotacao(self, rotulo: str, chave: str, casas: int, unidade: str = "") -> str:
        cotacao = dados_app.cotacao_mercado(chave, self.ctx.usar_internet)
        if cotacao is None:
            return self.ui.cartao_valor(rotulo, fmt.AUSENTE, detalhe="Fonte indisponível no momento",
                                        fonte="Yahoo Finance")
        valor = fmt.numero(cotacao.valor, casas) if casas == 0 else fmt.moeda(cotacao.valor, casas)
        return self.ui.cartao_valor(rotulo, valor, unidade, detalhe=f"{self.ui.variacao(cotacao.variacao_pct)} no dia",
                                    fonte=f"Yahoo Finance · {self.ui.origem(cotacao.origem, cotacao.obtido_em)}")

    def _cartao_dolar(self) -> str:
        """Dólar comercial pelo Yahoo Finance; na falta dele, a PTAX do Banco Central."""
        if dados_app.cotacao_mercado("dolar", self.ctx.usar_internet) is not None:
            return self._cartao_cotacao("Dólar comercial", "dolar", casas=2)
        ptax = dados_app.serie_bcb("dolar_ptax", self.ctx.usar_internet)
        if ptax is None or ptax.ultimo_valor is None:
            return self.ui.cartao_valor("Dólar comercial", fmt.AUSENTE, detalhe="Fonte indisponível no momento")
        anterior = float(ptax.dados["valor"].iloc[-2]) if len(ptax.dados) > 1 else None
        variacao = (ptax.ultimo_valor / anterior - 1) * 100 if anterior else None
        return self.ui.cartao_valor("Dólar (PTAX)", fmt.moeda(ptax.ultimo_valor),
                                    detalhe=f"{self.ui.variacao(variacao)} em {fmt.data_curta(ptax.ultima_data)}",
                                    fonte=f"Banco Central · {self.ui.origem(ptax.origem, ptax.obtido_em)}")

    def _cartao_selic(self, selic: SerieTemporal | None) -> str:
        if selic is None or selic.ultimo_valor is None:
            return self.ui.cartao_valor("Selic (meta)", fmt.AUSENTE, detalhe="Fonte indisponível no momento",
                                        fonte="Banco Central do Brasil")
        desde = self.ultima_mudanca(selic)
        return self.ui.cartao_valor("Selic (meta)", f"{fmt.numero(selic.ultimo_valor)}%", "a.a.",
                                    detalhe=f"em vigor desde {fmt.data_curta(desde)}",
                                    fonte=f"Banco Central · {self.ui.origem(selic.origem, selic.obtido_em)}")

    def _cartao_ipca(self, ipca: SerieTemporal | None) -> str:
        if ipca is None or ipca.ultimo_valor is None:
            return self.ui.cartao_valor("IPCA 12 meses", fmt.AUSENTE, detalhe="Fonte indisponível no momento",
                                        fonte="Banco Central do Brasil")
        anterior = float(ipca.dados["valor"].iloc[-2]) if len(ipca.dados) > 1 else None
        mudanca = "" if anterior is None else f" · {fmt.pontos_percentuais(ipca.ultimo_valor - anterior)} no mês"
        return self.ui.cartao_valor("IPCA 12 meses", f"{fmt.numero(ipca.ultimo_valor)}%",
                                    detalhe=f"até {fmt.mes_ano(ipca.ultima_data)}{mudanca}",
                                    fonte=f"Banco Central (IBGE) · {self.ui.origem(ipca.origem, ipca.obtido_em)}")

    @staticmethod
    def ultima_mudanca(serie: SerieTemporal) -> pd.Timestamp | None:
        """Data em que a série assumiu o valor atual (ex.: última decisão do Copom)."""
        dados = serie.dados
        if dados.empty:
            return None
        mudancas = dados[dados["valor"].diff().fillna(1) != 0]
        return mudancas["data"].iloc[-1]

    def _manchetes(self) -> None:
        self.ui.renderizar(self.ui.titulo_secao(
            "Manchetes da semana", "notícias recentes sobre Itaú, Bradesco, Banco do Brasil e Santander"))
        resultado = dados_app.manchetes(self.ctx.usar_internet)
        noticias = resultado.noticias
        if not noticias:
            self.ui.renderizar(self.ui.nota(
                "Não foi possível carregar as manchetes agora. Verifique a conexão com a internet "
                "ou tente de novo em alguns minutos."))
            return
        itens = "".join(self.ui.noticia(n.titulo, n.link, n.fonte, n.publicada_em, n.ticker) for n in noticias)
        self.ui.renderizar(self.ui.cartao_html(itens))
        st.caption(f"Fonte: Google Notícias (RSS), {self.ui.origem(resultado.origem, resultado.obtido_em)}. "
                   "As manchetes são apenas informativas e não entram nos modelos.")

    def _sinais_de_hoje(self, ultimos: list[dict]) -> None:
        if not ultimos:
            return
        nome_modelo = self.ctx.nome_modelo(self.ctx.modelo_principal)
        self.ui.renderizar(self.ui.titulo_secao(
            "Sinal de hoje para cada banco",
            f"modelo {nome_modelo}, pregão de {fmt.data_curta(ultimos[0]['data'])}; a confiança mostra o quanto "
            "o modelo prefere esse sinal aos outros dois, e não a chance de acerto"))
        cartoes = []
        for item in ultimos:
            confianca = "" if fmt.ausente(item["confianca"]) else (
                f"Confiança do modelo: {fmt.numero(item['confianca'], 0)}%")
            cartoes.append(self.ui.cartao_banco(item["ticker"], item["nome"], fmt.moeda(item["preco"]),
                                                self.ui.variacao(item["variacao"]), item["sinal"], confianca))
        self.ui.renderizar(self.ui.grade(cartoes, colunas=4))
        st.caption("Para entender o sinal de um banco em linguagem simples, abra o Painel Iniciante no menu ao lado.")
