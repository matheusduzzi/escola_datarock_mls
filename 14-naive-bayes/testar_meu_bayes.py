"""Testes da função categorica_log do meu_bayes.py.

Rode de dentro da pasta da aula:  python3 testar_meu_bayes.py
Os testes usam o canal da parte 10 da aula e casos pequenos feitos à mão, não as tentativas
do cartão: a leitura das tentativas continua com você.
"""
import math

import numpy as np
import pandas as pd

import meu_bayes
from bayes import carregar_compras, separar_compras
from meu_bayes import categorica_log

CONH, _ = separar_compras(carregar_compras())
CANAL = CONH["canal"]
FRAUDE = CONH["fraude"]


def perto(a, b, tol=0.0005):
    return abs(a - b) < tol


def testar_canal_da_aula():
    r = categorica_log(CANAL, FRAUDE, pd.Series(["app", "site"]))
    esperado = {(0, 0): 3602 / 6561, (0, 1): 51 / 90, (1, 0): 2959 / 6561, (1, 1): 39 / 90}
    for (linha, classe), p in esperado.items():
        veio = math.exp(r.iloc[linha][classe])
        assert perto(veio, p), (f"P({['app', 'site'][linha]} | classe {classe}) deu {veio:.4f}; "
                                f"a parte 10 da aula tem {p:.4f}")


def testar_colunas_e_indice():
    novos = pd.Series(["site", "app", "app"], index=[70, 10, 42])
    r = categorica_log(CANAL, FRAUDE, novos)
    assert isinstance(r, pd.DataFrame), f"veio {type(r).__name__}; precisa ser um DataFrame"
    assert list(r.columns) == [0, 1], f"colunas {list(r.columns)}; precisam ser as classes 0 e 1, em ordem"
    assert list(r.index) == [70, 10, 42], f"índice {list(r.index)}; precisa ser o índice de novos"


def testar_soma_um_em_cada_classe():
    r = categorica_log(CANAL, FRAUDE, pd.Series(["app", "site"]))
    for classe in (0, 1):
        soma = float(np.exp(r[classe]).sum())
        assert perto(soma, 1.0, 1e-9), f"na classe {classe}, as categorias somam {soma:.6f}; precisam somar 1"


def testar_categoria_que_falta_numa_classe():
    r = categorica_log(pd.Series(["a", "a", "b"]), [0, 0, 1], pd.Series(["b"]))
    veio = math.exp(r.iloc[0][0])
    assert perto(veio, 0.25), f"P(b | classe 0) deu {veio:.4f}; com alpha = 1 é (0 + 1) ÷ (2 + 1 × 2) = 0,25"


def testar_conta_as_categorias_do_treino_inteiro():
    r = categorica_log(pd.Series(["a", "b", "c"]), [0, 0, 1], pd.Series(["c", "a"]))
    veio = math.exp(r.iloc[0][0])
    assert perto(veio, 0.2), (f"P(c | classe 0) deu {veio:.4f}; o treino tem 3 categorias, então é "
                              f"(0 + 1) ÷ (2 + 1 × 3) = 0,2")
    veio = math.exp(r.iloc[1][1])
    assert perto(veio, 0.25), f"P(a | classe 1) deu {veio:.4f}; é (0 + 1) ÷ (1 + 1 × 3) = 0,25"


def testar_alpha_meio():
    r = categorica_log(pd.Series(["a", "a", "b"]), [0, 0, 1], pd.Series(["a"]), alpha=0.5)
    veio = math.exp(r.iloc[0][0])
    assert perto(veio, 2.5 / 3), f"com alpha = 0,5, P(a | classe 0) deu {veio:.4f}; é (2 + 0,5) ÷ (2 + 0,5 × 2)"


def testar_alpha_invalido():
    for alpha in (0, -1, 0.0):
        try:
            categorica_log(CANAL, FRAUDE, pd.Series(["app"]), alpha=alpha)
        except ValueError:
            continue
        raise AssertionError(f"alpha = {alpha!r} passou; precisa parar com ValueError")


def testar_categoria_nova():
    try:
        categorica_log(CANAL, FRAUDE, pd.Series(["app", "telefone"]))
    except ValueError:
        return
    raise AssertionError("«telefone» não está no treino e passou; precisa parar com ValueError")


def testar_tamanhos_diferentes():
    try:
        categorica_log(pd.Series(["a", "b", "a"]), [0, 1], pd.Series(["a"]))
    except ValueError:
        return
    raise AssertionError("treino com 3 casos e y com 2 passou; precisa parar com ValueError")


def testar_nao_muda_o_que_recebeu():
    treino = CANAL.copy()
    y = FRAUDE.copy()
    novos = pd.Series(["site", "app"], index=[5, 9])
    antes = (treino.copy(), y.copy(), novos.copy())
    categorica_log(treino, y, novos)
    assert treino.equals(antes[0]) and y.equals(antes[1]) and novos.equals(antes[2]), \
        "a função mudou treino, y ou novos"


def testar_classe_certa():
    r = categorica_log(pd.Series(["a", "b", "b", "b"]), [1, 0, 0, 0], pd.Series(["a"]))
    p0, p1 = math.exp(r.iloc[0][0]), math.exp(r.iloc[0][1])
    assert perto(p0, 0.2) and perto(p1, 2 / 3), (f"P(a | 0) = {p0:.4f} e P(a | 1) = {p1:.4f}; são "
                                                 f"(0 + 1) ÷ (3 + 2) = 0,2 e (1 + 1) ÷ (1 + 2) = 0,667")


def testar_escrita_no_proprio_arquivo():
    assert categorica_log.__module__ == "meu_bayes", \
        f"categorica_log veio de {categorica_log.__module__}; escreva no meu_bayes.py"
    assert meu_bayes.categorica_log is categorica_log


if __name__ == "__main__":
    import sys

    testes = [(nome, f) for nome, f in sorted(globals().items()) if nome.startswith("testar_")]
    falhas = 0
    for nome, teste in testes:
        try:
            teste()
        except NotImplementedError:
            raise
        except AssertionError as erro:
            falhas += 1
            print(f"FALHOU {nome}: {erro}")
        except Exception as erro:
            falhas += 1
            print(f"FALHOU {nome}: {type(erro).__name__}: {erro}")
    if falhas:
        print(f"{falhas} de {len(testes)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(testes)} testes passaram")
