"""Tarefa 3 do diagnóstico: padronizar o valor dos pedidos de abril.

Leia o programa inteiro antes de rodar:

    python3 tarefa3.py

Os 30 primeiros pedidos são o treino e os 10 últimos, da última semana de abril, são o teste.
O modelo vai ser usado em maio.
"""
import pandas as pd

df = pd.read_csv("dados/pedidos_abril.csv")

media = df["valor"].mean()
desvio = df["valor"].std()
df["valor_padronizado"] = (df["valor"] - media) / desvio

treino = df.iloc[:30]
teste = df.iloc[30:]

print("média usada:", round(media, 2))
print("desvio usado:", round(desvio, 2))
print("média padronizada no treino:", round(treino["valor_padronizado"].mean(), 3))
print("média padronizada no teste:", round(teste["valor_padronizado"].mean(), 3))
