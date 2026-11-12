"""Os modelos e as métricas da aula «Métricas e prevalência».

Cancelamento: o mesmo modelo da aula 07, treinado em janeiro a março e medido em abril e maio.
Fraude: as compras com cartão, com a mesma divisão pela data. Junho é o teste nas duas bases e
sai logo na leitura.

As métricas recebem a nota de cada linha (de 0 a 1, ou qualquer número em que maior quer dizer
mais risco) e o rótulo (1 para cancelado ou fraude, 0 para o resto).
"""
import numpy as np
import pandas as pd

FIM_TREINO = "2026-03-31"
FIM_VALIDACAO = "2026-05-31"
PENALIDADE = 1.0
PASSOS = 30

PEDIDOS = "../dados/loja/pedidos_export_completo.csv"
NUMERICAS = ["valor", "prazo_prometido_dias", "pedidos_anteriores", "cliente_novo"]
CATEGORIAS = ["categoria", "forma_pagamento", "canal"]

COMPRAS = "../dados/fraude/transacoes.csv"
COLUNAS_FRAUDE = ["valor", "itens", "idade_conta_dias", "tentativas_cartao",
                  "distancia_entrega_km", "hora", "app"]
MADRUGADA = [23, 0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------- o ajuste
def _newton(m, y, penalidade, passos):
    coef = np.zeros(m.shape[1])
    pen = np.eye(m.shape[1]) * penalidade
    pen[0, 0] = 0
    for _ in range(passos):
        p = 1 / (1 + np.exp(-(m @ coef)))
        grad = m.T @ (p - y) + pen @ coef
        hess = m.T @ (m * (p * (1 - p))[:, None]) + pen
        coef = coef - np.linalg.solve(hess, grad)
    return coef


def treinar(x, y):
    m = np.c_[np.ones(len(x)), x]
    with np.errstate(all="ignore"):
        return _newton(m, np.asarray(y, dtype=float), PENALIDADE, PASSOS)


def nota(coef, x):
    with np.errstate(all="ignore"):
        return 1 / (1 + np.exp(-(np.c_[np.ones(len(x)), x] @ coef)))


# ---------------------------------------------------------------- as duas bases
def cancelamento():
    """Retorna a nota e o rótulo dos 795 pedidos de abril e maio, e o rótulo do treino."""
    base = pd.read_csv(PEDIDOS, parse_dates=["data_pedido"])
    base = base[base["data_pedido"] <= pd.Timestamp(FIM_VALIDACAO)]          # junho sai aqui
    treino = base[base["data_pedido"] <= pd.Timestamp(FIM_TREINO)]
    valid = base[base["data_pedido"] > pd.Timestamp(FIM_TREINO)]
    media, desvio = treino[NUMERICAS].mean(), treino[NUMERICAS].std(ddof=0)
    categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}

    def matriz(p):
        x = (p[NUMERICAS] - media) / desvio
        for c in CATEGORIAS:
            for v in categorias[c]:
                x[f"{c}_{v}"] = (p[c] == v).astype(float)
        return x.to_numpy(dtype=float)

    coef = treinar(matriz(treino), treino["cancelado"])
    return nota(coef, matriz(valid)), valid["cancelado"].to_numpy(dtype=int), treino["cancelado"].to_numpy(dtype=int)


def compras():
    base = pd.read_csv(COMPRAS, parse_dates=["data_hora"])
    base = base[base["data_hora"] < pd.Timestamp(FIM_VALIDACAO) + pd.Timedelta(days=1)]   # junho sai aqui
    base = base.assign(app=(base["canal"] == "app").astype(int),
                       madrugada=base["hora"].isin(MADRUGADA).astype(int))
    treino = base[base["data_hora"] < pd.Timestamp(FIM_TREINO) + pd.Timedelta(days=1)]
    valid = base[base["data_hora"] >= pd.Timestamp(FIM_TREINO) + pd.Timedelta(days=1)]
    return treino, valid


def modelo_fraude(treino, valid, colunas):
    """Treina a logística nas colunas pedidas, na escala do treino, e dá nota às compras de abril e maio."""
    media, desvio = treino[colunas].mean(), treino[colunas].std(ddof=0)
    coef = treinar(((treino[colunas] - media) / desvio).to_numpy(), treino["fraude"])
    return nota(coef, ((valid[colunas] - media) / desvio).to_numpy())


# ---------------------------------------------------------------- as métricas
def matriz(nota, y, corte):
    """VP, FP, FN, VN da regra "marca quem tem nota maior ou igual ao corte"."""
    nota, y = np.asarray(nota), np.asarray(y)
    marca = nota >= corte
    return (int((marca & (y == 1)).sum()), int((marca & (y == 0)).sum()),
            int((~marca & (y == 1)).sum()), int((~marca & (y == 0)).sum()))


def no_corte(vp, fp, fn, vn):
    """Precisão, revocação, F1 e taxa de falso positivo. A precisão sem marcação nenhuma é None."""
    precisao = vp / (vp + fp) if vp + fp else None
    revocacao = vp / (vp + fn)
    f1 = 2 * vp / (2 * vp + fp + fn)
    falso_positivo = fp / (fp + vn)
    return precisao, revocacao, f1, falso_positivo


def topo(nota, y, k):
    """As k maiores notas: VP, FP, FN, VN. Empate na fronteira é desfeito pela ordem das linhas."""
    nota, y = np.asarray(nota), np.asarray(y)
    ordem = np.argsort(-nota, kind="mergesort")
    marca = np.zeros(len(y), dtype=bool)
    marca[ordem[:k]] = True
    return (int((marca & (y == 1)).sum()), int((marca & (y == 0)).sum()),
            int((~marca & (y == 1)).sum()), int((~marca & (y == 0)).sum()))


def acerto_do_par(nota, y):
    """A área sob a curva ROC: em quantos pares (positivo, negativo) o positivo tem nota maior."""
    posto = pd.Series(np.asarray(nota, dtype=float)).rank().to_numpy()
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((posto[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks(nota, y):
    """A maior distância entre revocação e taxa de falso positivo, e o corte onde ela acontece."""
    nota, y = np.asarray(nota), np.asarray(y)
    melhor, onde = -1.0, None
    for c in np.unique(nota):
        vp, fp, fn, vn = matriz(nota, y, c)
        d = vp / (vp + fn) - fp / (fp + vn)
        if d > melhor:
            melhor, onde = d, c
    return float(melhor), float(onde)


def precisao_media(nota, y):
    """A área sob a curva PR em degraus: a média da precisão no ponto de cada positivo.

    Notas empatadas entram juntas, como no average_precision_score do scikit-learn.
    """
    nota, y = np.asarray(nota, dtype=float), np.asarray(y)
    total = 0.0
    revocacao_antes = 0.0
    n1 = int((y == 1).sum())
    for c in np.unique(nota)[::-1]:
        vp, fp, fn, vn = matriz(nota, y, c)
        revocacao = vp / n1
        total += (revocacao - revocacao_antes) * vp / (vp + fp)
        revocacao_antes = revocacao
    return total


def precisao_com_outra_mistura(revocacao, falso_positivo, prevalencia):
    """A precisão da mesma regra numa base com outra parte de positivos."""
    acertos = revocacao * prevalencia
    return acertos / (acertos + falso_positivo * (1 - prevalencia))


# ---------------------------------------------------------------- números em português
def num(x, casas=1):
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, casas=1):
    return "-" if x is None else num(100 * x, casas) + "%"
