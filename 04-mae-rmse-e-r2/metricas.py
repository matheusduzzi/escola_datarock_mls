"""Mede o erro de uma regressão nos pedidos da Loja Exemplo: MAE, RMSE, R², R² ajustado e erro médio.

Usa o NumPy e o pandas, e o regressao.py da aula «Valor do pedido», que está nesta mesma pasta.
Cada métrica é uma linha de NumPy, para você ver a conta. O scikit-learn tem as mesmas três primeiras
em sklearn.metrics (mean_absolute_error, root_mean_squared_error e r2_score); o R² ajustado ele não tem.

    from metricas import dividir_por_data, prever, mae, rmse, r2, r2_ajustado, erro_medio, medir

O erro de cada pedido é sempre o valor pago menos o previsto, como o resíduo da aula 01.
"""
import numpy as np
import pandas as pd

from regressao import ajustar, matriz

CORTE = "2026-05-01"  # treino: janeiro a abril; teste: maio e junho


def _vetores(real, previsto):
    """Transforma em arrays e para se os tamanhos não batem, se está vazio ou se tem valor vazio."""
    real = np.asarray(real, dtype=float).ravel()
    previsto = np.asarray(previsto, dtype=float).ravel()
    if real.size == 0:
        raise ValueError("não há pedidos para medir")
    if real.size != previsto.size:
        raise ValueError(f"{real.size} valores reais e {previsto.size} previstos: os tamanhos precisam ser iguais")
    if np.isnan(real).any() or np.isnan(previsto).any():
        raise ValueError("há valor vazio no real ou no previsto")
    return real, previsto


def mae(real, previsto):
    """Erro absoluto médio: a média dos erros sem o sinal, em reais."""
    real, previsto = _vetores(real, previsto)
    return float(np.mean(np.abs(real - previsto)))


def rmse(real, previsto):
    """Raiz do erro quadrático médio: eleva cada erro ao quadrado, tira a média e volta para reais com a raiz."""
    real, previsto = _vetores(real, previsto)
    return float(np.sqrt(np.mean((real - previsto) ** 2)))


def erro_medio(real, previsto):
    """Média dos erros com o sinal. Positivo: o modelo previu menos do que os clientes pagaram."""
    real, previsto = _vetores(real, previsto)
    return float(np.mean(real - previsto))


def r2(real, previsto):
    """1 menos a soma dos quadrados dos erros dividida pela soma dos quadrados em volta da média do próprio real."""
    real, previsto = _vetores(real, previsto)
    if real.size < 2:
        raise ValueError("R² precisa de pelo menos dois pedidos")
    em_volta = np.sum((real - real.mean()) ** 2)
    if em_volta == 0:
        raise ValueError("todos os valores reais são iguais: o R² não tem com o que comparar")
    return float(1 - np.sum((real - previsto) ** 2) / em_volta)


def r2_ajustado(r2_valor, n, k):
    """R² ajustado pelo número de colunas k (sem contar o intercepto), com n pedidos."""
    if k < 0 or n - k - 1 <= 0:
        raise ValueError(f"{n} pedidos e {k} colunas: o R² ajustado precisa de mais pedidos que colunas mais um")
    return float(1 - (1 - r2_valor) * (n - 1) / (n - k - 1))


def dividir_por_data(dados, corte=CORTE):
    """Treino com os pedidos antes do corte e teste com os pedidos a partir dele."""
    treino = dados[dados["data_pedido"] < corte]
    teste = dados[dados["data_pedido"] >= corte]
    if len(treino) == 0 or len(teste) == 0:
        raise ValueError(f"o corte {corte} deixa o treino ou o teste sem pedidos")
    return treino, teste


def prever(ajuste, dados, colunas):
    """Aplica os coeficientes de um ajuste a outros pedidos, com as mesmas colunas do ajuste."""
    X = matriz(dados, colunas)
    if list(X.columns) != list(ajuste.coeficientes.index):
        raise ValueError("as colunas não são as mesmas do ajuste")
    return pd.Series(X.to_numpy(float).dot(ajuste.coeficientes.to_numpy(float)), index=dados.index)


def medir(real, previsto):
    """As quatro medidas de uma vez, num dicionário."""
    return {"MAE": mae(real, previsto), "RMSE": rmse(real, previsto), "R²": r2(real, previsto),
            "erro médio": erro_medio(real, previsto)}


def com_colunas_sorteadas(dados, quantas, sorteio=2026):
    """Uma cópia dos pedidos com `quantas` colunas de números sorteados, sem relação nenhuma com o valor."""
    rng = np.random.default_rng(sorteio)
    nomes = [f"sorteada_{i:02d}" for i in range(1, quantas + 1)]
    sorteadas = pd.DataFrame(rng.normal(size=(len(dados), quantas)), index=dados.index, columns=nomes)
    return pd.concat([dados, sorteadas], axis=1), nomes


def treinar_e_medir(treino, teste, colunas):
    """Ajusta no treino e mede no treino e no teste. Retorna o ajuste e as duas medidas."""
    ajuste = ajustar(treino, colunas)
    no_treino = medir(treino["valor"], ajuste.previsto)
    no_teste = medir(teste["valor"], prever(ajuste, teste, colunas))
    return ajuste, no_treino, no_teste
