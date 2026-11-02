"""
Clientes das fontes externas de dados contextuais.

Todas as integrações herdam de ClienteHTTP, que aplica tempo-limite às
requisições e guarda em disco a última resposta válida. Se a internet (ou a
API) estiver indisponível, a interface continua funcionando com a cópia
local e informa ao usuário a data da informação exibida.
"""

from econom_ia.integracoes.bcb import ClienteBCB
from econom_ia.integracoes.focus import ClienteFocus, ExpectativasSelic
from econom_ia.integracoes.fred import ClienteFRED
from econom_ia.integracoes.http import ClienteHTTP, ErroIntegracao, Resposta, SerieTemporal
from econom_ia.integracoes.mercado import ClienteMercado, Cotacao
from econom_ia.integracoes.noticias import ClienteNoticias, Manchetes, Noticia
from econom_ia.integracoes.polymarket import ClientePolymarket, MercadoPrevisao

__all__ = [
    "ClienteBCB",
    "ClienteFRED",
    "ClienteFocus",
    "ClienteHTTP",
    "ClienteMercado",
    "ClienteNoticias",
    "ClientePolymarket",
    "Cotacao",
    "ErroIntegracao",
    "ExpectativasSelic",
    "Manchetes",
    "MercadoPrevisao",
    "Noticia",
    "Resposta",
    "SerieTemporal",
]
