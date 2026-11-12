"""Primeira olhada nos pedidos e nas compras com cartão da Loja Exemplo.

Rode de dentro desta pasta:

    python3 primeira_olhada.py

São três contas: o valor médio com e sem cupom, duas regras para prever o valor
de maio e junho com o que se sabia até abril, e a regra "nunca é fraude".
"""
import numpy as np
import pandas as pd


def reais(x):
    return f"{x:.2f}".replace(".", ",")


def pct(parte, todo):
    return f"{100 * parte / todo:.1f}".replace(".", ",") + "%"


pedidos = pd.read_csv("../dados/valor/pedidos_valor.csv", parse_dates=["data_pedido"])
compras = pd.read_csv("../dados/fraude/transacoes.csv")

print("1. Valor médio com e sem cupom, em reais")
grupos = [("todos", pedidos), ("recorrentes", pedidos[pedidos["cliente_novo"] == 0]),
          ("novos", pedidos[pedidos["cliente_novo"] == 1])]
for nome, g in grupos:
    com = g[g["cupom"] == 1]["valor"]
    sem = g[g["cupom"] == 0]["valor"]
    print(f"   {nome:<12} com cupom {reais(com.mean()):>7} ({len(com)})"
          f"   sem cupom {reais(sem.mean()):>7} ({len(sem)})")
com_cupom = pedidos[pedidos["cupom"] == 1]
print("   pedidos com cupom que são de cliente novo:",
      pct(com_cupom["cliente_novo"].sum(), len(com_cupom)))

print("2. Prever o valor de maio e junho com o que se sabia até abril")
treino = pedidos[pedidos["data_pedido"] < "2026-05-01"]
teste = pedidos[pedidos["data_pedido"] >= "2026-05-01"]
print(f"   jan a abr: {len(treino)} pedidos para ajustar; mai e jun: {len(teste)} para medir")
media = treino["valor"].mean()
erro_media = (teste["valor"] - media).abs().mean()
print(f"   todo pedido vale a média de jan a abr ({reais(media)}): erra {reais(erro_media)} por pedido")
b, a = np.polyfit(treino["itens"], treino["valor"], 1)
erro_reta = (teste["valor"] - (a + b * teste["itens"])).abs().mean()
print(f"   reta pelos itens ({reais(a)} + {reais(b)} por item): erra {reais(erro_reta)} por pedido")
grandes = treino["itens"] >= 40
b2, a2 = np.polyfit(treino.loc[~grandes, "itens"], treino.loc[~grandes, "valor"], 1)
print(f"   a mesma reta sem os {int(grandes.sum())} pedidos de jan a abr com 40 itens ou mais:"
      f" {reais(a2)} + {reais(b2)} por item")

print("3. Compras com cartão")
fraudes = int(compras["fraude"].sum())
print(f"   {len(compras)} compras, {fraudes} fraudes ({pct(fraudes, len(compras))})")
print(f"   nunca é fraude: acerta {len(compras) - fraudes} de {len(compras)}"
      f" ({pct(len(compras) - fraudes, len(compras))}), pega 0 de {fraudes} fraudes")
