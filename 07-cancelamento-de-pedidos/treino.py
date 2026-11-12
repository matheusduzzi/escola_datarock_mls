"""A regressão logística da aula «Cancelamento de pedidos», treinada pelo gradiente da log-loss.

É a conta da aula 05, «Logística e log-loss», com todas as colunas que o pipeline entrega no lugar
de uma coluna só. A penalidade pequena é a mesma do classificador pronto do curso anterior, para
os dois chegarem ao mesmo modelo. A aula 10, «Ridge, Lasso e Elastic Net», explica a penalidade.

treinar(x, y, taxa, passos) retorna {"colunas", "pesos"}, no mesmo formato do classificador
pronto. "pesos" começa pelo intercepto.
nota(modelo, x): a nota de 0 a 1 de cada linha. Para se as colunas não forem as do treino.
log_loss(nota, y): a perda média, como na aula 05.
"""
import numpy as np
import pandas as pd

PENALIDADE = 1.0      # a mesma do classificador pronto
TAXA = 4.0
PASSOS = 5000


def sigmoide(score):
    return 1 / (1 + np.exp(-score))


def log_loss(nota, y):
    nota = np.asarray(nota, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(all="ignore"):
        return float(np.mean(-(y * np.log(nota) + (1 - y) * np.log(1 - nota))))


def treinar(x, y, taxa=TAXA, passos=PASSOS, penalidade=PENALIDADE):
    m = np.c_[np.ones(len(x)), np.asarray(x, dtype=float)]
    y = np.asarray(y, dtype=float)
    n = len(y)
    pen = np.full(m.shape[1], penalidade)
    pen[0] = 0                                    # o intercepto não paga penalidade
    pesos = np.zeros(m.shape[1])                  # começa com nota 0,5 para todo pedido
    with np.errstate(all="ignore"):
        for _ in range(passos):
            nota_ = sigmoide(m @ pesos)
            gradiente = (m.T @ (nota_ - y) + pen * pesos) / n   # o gradiente da aula 05, com a penalidade
            pesos = pesos - taxa * gradiente
    return {"colunas": list(x.columns), "pesos": pesos}


def nota(modelo, x):
    if list(x.columns) != list(modelo["colunas"]):
        raise ValueError("as colunas não são as do treino do modelo")
    m = np.c_[np.ones(len(x)), np.asarray(x, dtype=float)]
    with np.errstate(all="ignore"):
        return pd.Series(sigmoide(m @ np.asarray(modelo["pesos"], dtype=float)), index=x.index)
