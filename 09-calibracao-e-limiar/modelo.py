"""O modelo de cancelamento e as medidas da aula «Calibração e limiar».

ler(caminho): lê o export dos pedidos e tira junho logo na leitura. Junho é o teste e fica fechado.
preparar(base): separa janeiro a março (treino), abril e maio. Média, desvio e categorias saem só
do treino. Só entram colunas que existem na hora do pedido.
treinar(x, y, peso_cancelado): regressão logística pelo método de Newton, com uma penalidade
pequena. Com peso_cancelado maior que 1, cada cancelado conta mais na conta do erro.
nota(modelo, x): a nota de 0 a 1 de cada pedido.
acerto_do_par(nota, y): sorteando um cancelado e um entregue, em quantos pares o cancelado tem a
nota maior. Só olha a ordem das notas.
ajustar_escala(nota, y) e corrigir(escala, nota): a correção de Platt, uma regressão logística
com uma coluna só, o logit da nota. Ela muda a escala e mantém a ordem.
"""
import numpy as np
import pandas as pd

FIM_TREINO = "2026-03-31"
FIM_ABRIL = "2026-04-30"
FIM_VALIDACAO = "2026-05-31"
NUMERICAS = ["valor", "prazo_prometido_dias", "pedidos_anteriores", "cliente_novo"]
CATEGORIAS = ["categoria", "forma_pagamento", "canal"]
PENALIDADE = 1.0
PASSOS = 30


def ler(caminho):
    base = pd.read_csv(caminho, parse_dates=["data_pedido"])
    return base[base["data_pedido"] <= pd.Timestamp(FIM_VALIDACAO)]    # junho sai aqui


def preparar(base):
    treino = base[base["data_pedido"] <= pd.Timestamp(FIM_TREINO)]
    media = treino[NUMERICAS].mean()
    desvio = treino[NUMERICAS].std(ddof=0)
    categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}

    def matriz(p):
        x = (p[NUMERICAS] - media) / desvio
        for c in CATEGORIAS:
            for v in categorias[c]:
                x[f"{c}_{v}"] = (p[c] == v).astype(float)
        return x.to_numpy(dtype=float), p["cancelado"].to_numpy(dtype=int), p["id_pedido"].to_numpy()

    abril = base[(base["data_pedido"] > pd.Timestamp(FIM_TREINO)) & (base["data_pedido"] <= pd.Timestamp(FIM_ABRIL))]
    maio = base[base["data_pedido"] > pd.Timestamp(FIM_ABRIL)]
    return {"treino": matriz(treino), "abril": matriz(abril), "maio": matriz(maio)}


def _newton(m, y, peso, penalidade, passos):
    coef = np.zeros(m.shape[1])
    pen = np.eye(m.shape[1]) * penalidade
    pen[0, 0] = 0
    for _ in range(passos):
        p = 1 / (1 + np.exp(-(m @ coef)))
        grad = m.T @ (peso * (p - y)) + pen @ coef
        hess = m.T @ (m * (peso * p * (1 - p))[:, None]) + pen
        coef = coef - np.linalg.solve(hess, grad)
    return coef


def treinar(x, y, peso_cancelado=1.0):
    y = np.asarray(y, dtype=float)
    peso = np.where(y == 1, peso_cancelado, 1.0)
    m = np.c_[np.ones(len(x)), x]
    with np.errstate(all="ignore"):
        return _newton(m, y, peso, PENALIDADE, PASSOS)


def nota(coef, x):
    with np.errstate(all="ignore"):
        return 1 / (1 + np.exp(-(np.c_[np.ones(len(x)), x] @ coef)))


def acerto_do_par(nota, y):
    nota = pd.Series(np.asarray(nota, dtype=float))
    y = np.asarray(y)
    posto = nota.rank().to_numpy()
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((posto[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def ajustar_escala(nota, y):
    """Platt: nota corrigida = 1 / (1 + exp(-(a + b × logit(nota)))). Retorna (a, b)."""
    m = np.c_[np.ones(len(nota)), logit(nota)]
    with np.errstate(all="ignore"):
        a, b = _newton(m, np.asarray(y, dtype=float), np.ones(len(nota)), 0.0, 50)
    return float(a), float(b)


def corrigir(escala, nota):
    a, b = escala
    return 1 / (1 + np.exp(-(a + b * logit(nota))))


def corte_equivalente(escala, corte):
    """A nota sem correção que vira exatamente `corte` depois da correção."""
    a, b = escala
    z = (np.log(corte / (1 - corte)) - a) / b
    return float(1 / (1 + np.exp(-z)))


def num(x, casas=1):
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, casas=1):
    return num(100 * x, casas) + "%"


def reais(x):
    return "R$ " + num(x, 0)
