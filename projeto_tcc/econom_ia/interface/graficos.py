"""
Gráficos interativos (Plotly) usados nas telas.

Convenções visuais: linhas finas, grade discreta, legenda sempre presente
quando há mais de uma série (e ausente quando há uma só), um único eixo
vertical por gráfico e cores das séries em ordem fixa. Previsões aparecem
tracejadas, pois são projeções. Números seguem o padrão brasileiro (vírgula
decimal) e as datas o formato dia/mês/ano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from econom_ia.config import ATIVOS, CLASSES
from econom_ia.interface.tema import Paleta

NOMES_CONJUNTOS = {"teste": "Teste", "oos": "Out-of-sample (SANB11)", "prospectivo": "Prospectivo"}

# Formato das datas no eixo horizontal conforme o espaçamento entre as marcas
# (em milissegundos): dia/mês até ~3 semanas, mês/ano até ~10 meses e ano acima
# disso. Os nomes dos meses são evitados porque o Plotly os escreve em inglês.
FORMATOS_DATA = [
    dict(dtickrange=[None, 2.0e9], value="%d/%m"),
    dict(dtickrange=[2.0e9, 2.6e10], value="%m/%Y"),
    dict(dtickrange=[2.6e10, None], value="%Y"),
]


class FabricaGraficos:
    """Cria as figuras Plotly com o estilo do Econom-IA."""

    # ── Estilo comum ───────────────────────────────────────────────────────
    @staticmethod
    def _estilizar(fig: go.Figure, altura: int = 420, titulo_y: str | None = None,
                   hover: str = "x unified", datas: bool = False) -> go.Figure:
        fig.update_layout(
            template="plotly_dark",
            separators=",.",  # vírgula decimal e ponto de milhar
            height=altura,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", size=14),
            hovermode=hover,
            hoverlabel=dict(bgcolor=Paleta.CARTAO, bordercolor=Paleta.BORDA_FORTE,
                            font=dict(color=Paleta.TEXTO, size=14)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=14)),
            showlegend=len(fig.data) > 1,
            margin=dict(l=10, r=10, t=44, b=10),
        )
        fig.update_xaxes(gridcolor=Paleta.BORDA, linecolor=Paleta.BORDA, color=Paleta.TEXTO_SECUNDARIO,
                         showgrid=False, zeroline=False)
        fig.update_yaxes(gridcolor=Paleta.BORDA, linecolor=Paleta.BORDA, color=Paleta.TEXTO_SECUNDARIO,
                         zeroline=False, title=titulo_y)
        if datas:
            fig.update_xaxes(tickformatstops=FORMATOS_DATA, hoverformat="%d/%m/%Y")
        return fig

    # ── Preços ─────────────────────────────────────────────────────────────
    @classmethod
    def preco_com_sinais(cls, dados: pd.DataFrame, previsao: pd.DataFrame | None, ticker: str,
                         pregoes_visiveis: int | None = 126, alertas: pd.DataFrame | None = None,
                         mostrar_bandas: bool = True) -> go.Figure:
        """Preço, média de 21 pregões, Bandas de Bollinger, alertas e previsão com faixa de 80%."""
        visivel = dados if pregoes_visiveis is None else dados.tail(pregoes_visiveis)
        cor = ATIVOS[ticker].cor if ticker in ATIVOS else Paleta.DESTAQUE
        fig = go.Figure()
        if mostrar_bandas and "bb_upper" in visivel:
            fig.add_trace(go.Scatter(
                x=list(visivel["data"]) + list(visivel["data"][::-1]),
                y=list(visivel["bb_upper"]) + list(visivel["bb_lower"][::-1]),
                fill="toself", fillcolor="rgba(148,163,184,0.10)", line=dict(width=0), mode="lines",
                name="Bandas de Bollinger (20)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=visivel["data"], y=visivel["sma_21"], mode="lines",
                                 name="Média móvel de 21 pregões", line=dict(color=Paleta.TEXTO_FRACO, width=1.4),
                                 hovertemplate="R$ %{y:.2f}"))
        fig.add_trace(go.Scatter(x=visivel["data"], y=visivel["close"], mode="lines",
                                 name="Preço de fechamento", line=dict(color=cor, width=2),
                                 hovertemplate="R$ %{y:.2f}"))
        if alertas is not None and not alertas.empty:
            recentes = alertas[alertas["data"] >= visivel["data"].min()]
            for tipo, simbolo, nome in (("compra", "triangle-up", "Alerta de compra"),
                                        ("venda", "triangle-down", "Alerta de venda")):
                pontos = recentes[recentes["tipo"] == tipo]
                if pontos.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=pontos["data"], y=pontos["preco"], mode="markers", name=nome,
                    marker=dict(symbol=simbolo, size=11, color=Paleta.cor_sinal(tipo),
                                line=dict(color=Paleta.FUNDO, width=1.5)),
                    hovertemplate=f"{nome}: R$ %{{y:.2f}}<extra></extra>"))
        if previsao is not None and not previsao.empty:
            cls._adicionar_previsao(fig, visivel.iloc[-1], previsao)
        return cls._estilizar(fig, altura=470, titulo_y="Preço (R$)", datas=True)

    @staticmethod
    def _adicionar_previsao(fig: go.Figure, ultimo: pd.Series, previsao: pd.DataFrame,
                            nome: str = "Previsão LSTM") -> None:
        """Linha tracejada da previsão, partindo do último preço, com a faixa de 80%."""
        datas = [ultimo["data"], *previsao["data"]]
        fig.add_trace(go.Scatter(
            x=datas + datas[::-1],
            y=[ultimo["close"], *previsao["lim_sup"]] + [ultimo["close"], *previsao["lim_inf"]][::-1],
            fill="toself", fillcolor="rgba(201,133,0,0.18)", line=dict(width=0), mode="lines",
            name="Faixa provável (80%)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=datas, y=[ultimo["close"], *previsao["previsao"]], mode="lines+markers", name=nome,
            line=dict(color=Paleta.PREVISAO, width=2, dash="dash"), marker=dict(size=8),
            hovertemplate="Previsto: R$ %{y:.2f}"))

    @classmethod
    def preco_simples(cls, dados: pd.DataFrame, previsao: pd.DataFrame | None, ticker: str,
                      pregoes_visiveis: int | None = 126) -> go.Figure:
        """Versão enxuta para o Painel Iniciante: preço e estimativa."""
        visivel = dados if pregoes_visiveis is None else dados.tail(pregoes_visiveis)
        cor = ATIVOS[ticker].cor if ticker in ATIVOS else Paleta.DESTAQUE
        fig = go.Figure(go.Scatter(x=visivel["data"], y=visivel["close"], mode="lines", name="Preço da ação",
                                   line=dict(color=cor, width=2.4), hovertemplate="R$ %{y:.2f}"))
        if previsao is not None and not previsao.empty:
            cls._adicionar_previsao(fig, visivel.iloc[-1], previsao, nome="Estimativa")
        return cls._estilizar(fig, altura=400, titulo_y="Preço (R$)", datas=True)

    @classmethod
    def desempenho_setor(cls, dados: pd.DataFrame, pregoes: int = 126) -> go.Figure:
        """Evolução dos quatro bancos com base 100 no início do período."""
        fig = go.Figure()
        for ticker, ativo in ATIVOS.items():
            serie = dados[dados["ticker"] == ticker].tail(pregoes)
            if serie.empty:
                continue
            base_100 = serie["close"] / serie["close"].iloc[0] * 100
            fig.add_trace(go.Scatter(x=serie["data"], y=base_100, mode="lines", name=ticker,
                                     line=dict(color=ativo.cor, width=2), hovertemplate="%{y:.1f}"))
        fig.add_hline(y=100, line=dict(color=Paleta.BORDA_FORTE, width=1))
        return cls._estilizar(fig, altura=330, titulo_y="Base 100", datas=True)

    # ── Macroeconomia ──────────────────────────────────────────────────────
    @classmethod
    def series_temporais(cls, series: dict[str, pd.DataFrame], titulo_y: str, altura: int = 340,
                         degraus: bool = False) -> go.Figure:
        """Várias séries na mesma unidade (um único eixo)."""
        fig = go.Figure()
        for i, (nome, dados) in enumerate(series.items()):
            fig.add_trace(go.Scatter(x=dados["data"], y=dados["valor"], mode="lines", name=nome,
                                     line=dict(color=Paleta.SERIES[i % len(Paleta.SERIES)], width=2,
                                               shape="hv" if degraus else "linear"),
                                     hovertemplate="%{y:.2f}"))
        return cls._estilizar(fig, altura=altura, titulo_y=titulo_y, datas=True)

    @classmethod
    def trajetoria_selic(cls, reunioes: pd.DataFrame, selic_atual: float | None) -> go.Figure:
        """Mediana das projeções do Focus para cada reunião do Copom, com o intervalo mínimo a máximo."""
        rotulos = list(reunioes["reuniao"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rotulos + rotulos[::-1], y=list(reunioes["maximo"]) + list(reunioes["minimo"])[::-1],
            fill="toself", fillcolor="rgba(57,135,229,0.14)", line=dict(width=0), mode="lines",
            name="Intervalo das projeções (mínimo a máximo)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=rotulos, y=reunioes["mediana"], mode="lines+markers", name="Mediana das projeções",
            line=dict(color=Paleta.DESTAQUE, width=2, dash="dash"), marker=dict(size=8),
            customdata=reunioes[["minimo", "maximo", "respondentes"]].to_numpy(),
            hovertemplate="%{x}: mediana %{y:.2f}%<br>mínimo %{customdata[0]:.2f}% · máximo %{customdata[1]:.2f}%"
                          "<br>%{customdata[2]} instituições<extra></extra>"))
        if selic_atual is not None:
            fig.add_hline(y=selic_atual, line=dict(color=Paleta.ALERTA, width=1.2),
                          annotation_text=f"Selic atual: {selic_atual:.2f}%".replace(".", ","),
                          annotation_font_color=Paleta.ALERTA, annotation_position="top right")
        fig = cls._estilizar(fig, altura=360, titulo_y="Meta Selic (% ao ano)", hover="closest")
        fig.update_xaxes(title="Reunião do Copom", type="category")
        return fig

    @classmethod
    def probabilidades(cls, opcoes: list[tuple[str, float]]) -> go.Figure:
        """Probabilidade de cada resultado (barras horizontais, maior no topo)."""
        rotulos = [rotulo for rotulo, _ in opcoes][::-1]
        valores = [prob * 100 for _, prob in opcoes][::-1]
        fig = go.Figure(go.Bar(
            x=valores, y=rotulos, orientation="h", marker=dict(color=Paleta.DESTAQUE, cornerradius=4),
            text=[f"{v:.0f}%" for v in valores], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_xaxes(range=[0, 112], ticksuffix="%")
        return cls._estilizar(fig, altura=max(180, 56 * len(rotulos) + 50), hover="closest")

    # ── Desempenho dos modelos ─────────────────────────────────────────────
    @classmethod
    def comparacao_modelos(cls, linhas: list[dict], referencia: float | None) -> go.Figure:
        """
        F1-Score macro de cada modelo nos conjuntos de avaliação, com IC de 95%.

        Args:
            linhas: dicionários com modelo, conjunto, f1, ic_inferior e ic_superior.
            referencia: F1 do classificador aleatório estratificado no teste.
        """
        tabela = pd.DataFrame(linhas)
        fig = go.Figure()
        for i, conjunto in enumerate(["teste", "oos", "prospectivo"]):
            parte = tabela[tabela["conjunto"] == conjunto]
            if parte.empty:
                continue
            erro = None
            if parte["ic_inferior"].notna().all():
                erro = dict(type="data", symmetric=False, color=Paleta.TEXTO_SECUNDARIO, thickness=1.2, width=4,
                            array=parte["ic_superior"] - parte["f1"], arrayminus=parte["f1"] - parte["ic_inferior"])
            fig.add_trace(go.Bar(x=parte["modelo"], y=parte["f1"], name=NOMES_CONJUNTOS[conjunto],
                                 marker=dict(color=Paleta.SERIES[i], cornerradius=4), error_y=erro,
                                 hovertemplate="%{x}: F1 %{y:.3f}<extra></extra>"))
        if referencia is not None:
            cls._linha_referencia(fig, referencia, f"Aleatório no teste ({referencia:.3f})".replace(".", ","))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
        fig.update_yaxes(range=[0, max(0.5, float(tabela["ic_superior"].fillna(tabela["f1"]).max()) + 0.06)])
        return cls._estilizar(fig, altura=390, titulo_y="F1-Score macro", hover="closest")

    @classmethod
    def barras_simples(cls, rotulos: list[str], valores: list[float], titulo_y: str,
                       formato: str = "{:.3f}", erros: list[float] | None = None,
                       referencia: float | None = None, rotulo_referencia: str = "",
                       horizontal: bool = False, polaridade: bool = False) -> go.Figure:
        """
        Uma única série de barras, com linha de referência opcional.

        Args:
            polaridade: se verdadeiro, valores negativos ficam em cinza, para
                distinguir o que pesa a favor do que pesa contra.
        """
        texto = [formato.format(v).replace(".", ",") for v in valores]
        cores = ([Paleta.DESTAQUE if v >= 0 else Paleta.TEXTO_FRACO for v in valores]
                 if polaridade else Paleta.DESTAQUE)
        if horizontal:
            barra = go.Bar(x=valores, y=rotulos, orientation="h", text=texto, textposition="outside",
                           cliponaxis=False, marker=dict(color=cores, cornerradius=4),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>")
        else:
            barra = go.Bar(x=rotulos, y=valores, text=texto, textposition="outside", cliponaxis=False,
                           marker=dict(color=cores, cornerradius=4),
                           error_y=dict(type="data", array=erros, color=Paleta.TEXTO_SECUNDARIO,
                                        thickness=1.2, width=4) if erros else None,
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>")
        barra.showlegend = False
        fig = go.Figure(barra)
        if referencia is not None:
            cls._linha_referencia(fig, referencia, rotulo_referencia, vertical=horizontal)
        altura = max(260, 30 * len(rotulos) + 70) if horizontal else 330
        fig = cls._estilizar(fig, altura=altura, hover="closest")
        if horizontal:
            fig.update_xaxes(title=titulo_y, zeroline=True, zerolinecolor=Paleta.BORDA_FORTE)
            fig.update_yaxes(autorange="reversed", title=None)
        else:
            fig.update_yaxes(title=titulo_y)
        return fig

    @staticmethod
    def _linha_referencia(fig: go.Figure, valor: float, rotulo: str, vertical: bool = False) -> None:
        """
        Linha de referência (ex.: desempenho ao acaso) identificada na legenda.

        A legenda substitui a anotação sobre o gráfico, que acabava sobreposta
        aos rótulos das barras.
        """
        estilo = dict(color=Paleta.ALERTA, width=1.4)
        if vertical:
            fig.add_vline(x=valor, line=estilo)
        else:
            fig.add_hline(y=valor, line=estilo)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=estilo, name=rotulo, hoverinfo="skip"))

    @classmethod
    def barras_agrupadas(cls, categorias: list[str], series: dict[str, list[float]], titulo_y: str,
                         formato: str = "{:.3f}") -> go.Figure:
        """Duas ou mais séries de barras lado a lado, na mesma unidade."""
        fig = go.Figure()
        for i, (nome, valores) in enumerate(series.items()):
            fig.add_trace(go.Bar(x=categorias, y=valores, name=nome,
                                 marker=dict(color=Paleta.SERIES[i % len(Paleta.SERIES)], cornerradius=4),
                                 text=[formato.format(v).replace(".", ",") for v in valores],
                                 textposition="outside", cliponaxis=False,
                                 hovertemplate=f"{nome}: %{{y:.3f}}<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
        return cls._estilizar(fig, altura=340, titulo_y=titulo_y, hover="x unified")

    @classmethod
    def matriz_confusao(cls, matriz: list[list[int]]) -> go.Figure:
        """Matriz de confusão com contagens e proporção por classe real (linha)."""
        valores = np.array(matriz, dtype=float)
        proporcao = valores / np.maximum(valores.sum(axis=1, keepdims=True), 1)
        rotulos = [c.capitalize() for c in CLASSES]
        texto = [[f"{int(valores[i, j])}<br>{proporcao[i, j]:.0%}" for j in range(3)] for i in range(3)]
        fig = go.Figure(go.Heatmap(
            z=proporcao, x=rotulos, y=rotulos, text=texto, texttemplate="%{text}",
            textfont=dict(color=Paleta.TEXTO, size=14),
            colorscale=[[0, "#111a29"], [1, "#2f6fc0"]], zmin=0, zmax=1, xgap=2, ygap=2,
            showscale=False, hovertemplate="Real: %{y}<br>Previsto: %{x}<br>%{text}<extra></extra>"))
        fig = cls._estilizar(fig, altura=330, hover="closest")
        fig.update_xaxes(title="Classe prevista", side="bottom")
        fig.update_yaxes(title="Classe real", autorange="reversed")
        return fig

    @classmethod
    def curvas_capital(cls, curvas: dict, nomes: dict[str, str]) -> go.Figure:
        """Evolução de R$ 1 aplicado seguindo os sinais de cada modelo, versus comprar e manter."""
        datas = pd.to_datetime(curvas["datas"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datas, y=curvas["comprar_e_manter"], mode="lines", name="Comprar e manter",
                                 line=dict(color=Paleta.TEXTO_SECUNDARIO, width=1.6), hovertemplate="R$ %{y:.3f}"))
        for i, (chave, nome) in enumerate(nomes.items()):
            if chave in curvas:
                fig.add_trace(go.Scatter(x=datas, y=curvas[chave], mode="lines", name=nome,
                                         line=dict(color=Paleta.SERIES[i % 4], width=2),
                                         hovertemplate="R$ %{y:.3f}"))
        fig.add_hline(y=1, line=dict(color=Paleta.BORDA_FORTE, width=1))
        return cls._estilizar(fig, altura=390, titulo_y="Valor de R$ 1 aplicado", datas=True)

    @classmethod
    def previsto_vs_real(cls, tabela: pd.DataFrame, ticker: str, horizonte: int = 5) -> go.Figure:
        """
        Preço real "horizonte" pregões depois de cada data versus o previsto pela LSTM.

        Args:
            tabela: colunas data (quando a previsão foi feita), preco_real e preco_previsto.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tabela["data"], y=tabela["preco_real"], mode="lines",
                                 name=f"Preço real {horizonte} pregões depois",
                                 line=dict(color=ATIVOS[ticker].cor, width=2), hovertemplate="R$ %{y:.2f}"))
        fig.add_trace(go.Scatter(x=tabela["data"], y=tabela["preco_previsto"], mode="lines",
                                 name="Previsão da LSTM", line=dict(color=Paleta.PREVISAO, width=1.6, dash="dash"),
                                 hovertemplate="R$ %{y:.2f}"))
        fig = cls._estilizar(fig, altura=360, titulo_y="Preço (R$)", datas=True)
        fig.update_xaxes(title="Data em que a previsão foi feita")
        return fig
