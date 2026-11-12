"""Testes das funções kernel_rbf e decisao do meu_kernel.py.

Rode de dentro da pasta da aula:  python3 testar_meu_kernel.py
Os testes usam os números da aula (os dez pedidos da parte 1 e o par 5013 e 5024 da parte 6),
não os do exercício: eles mostram que o código faz a conta da aula, e a leitura do exercício
continua com você.
"""
import math

import numpy as np

import fronteiras as fr
import meu_kernel
from meu_kernel import decisao, kernel_rbf

PED = fr.carregar_pedidos()
MARCO, ABRIL = fr.carregar_entregas()
XM = MARCO[fr.MAPA].to_numpy(float)
YM = fr.sinal(MARCO["cancelado"])
GAMMA = 1 / (XM.shape[1] * XM.var())
X10 = PED.loc[fr.DEZ, fr.DUAS].to_numpy(float)
Y10 = fr.sinal(PED.loc[fr.DEZ, "cancelado"])
RETA = fr.treinar(X10, Y10, fr.k_linear, 1000.0)


def perto(a, b, tol=0.005):
    return abs(a - b) < tol


def testar_mesmo_ponto_vale_um():
    k = kernel_rbf(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]), 0.5)
    assert perto(k[0, 0], 1.0, 1e-12), f"um pedido com ele mesmo deu {k[0, 0]}; distância 0 dá exp(0) = 1"


def testar_distancia_um():
    for g in (1.0, 2.0):
        k = kernel_rbf(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), g)
        assert perto(k[0, 0], math.exp(-g), 1e-12), (
            f"a 1 km, com gamma {g}, deu {k[0, 0]:.4f}; exp(−{g} × 1²) é {math.exp(-g):.4f}")


def testar_quadrado_da_distancia():
    k = kernel_rbf(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]), 0.01)
    assert perto(k[0, 0], math.exp(-0.25), 1e-12), (
        f"de (0, 0) até (3, 4), com gamma 0,01, deu {k[0, 0]:.4f}; a distância é 5, o quadrado é 25, "
        f"e exp(−0,01 × 25) é {math.exp(-0.25):.4f}")


def testar_sinal_negativo():
    k = kernel_rbf(np.array([[0.0, 0.0]]), np.array([[2.0, 0.0], [8.0, 0.0]]), 0.1)
    assert k[0, 0] < 1 and k[0, 1] < k[0, 0], "o kernel precisa cair com a distância: mais longe, mais perto de 0"


def testar_formato():
    A = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    B = np.array([[0.0, 1.0], [2.0, 2.0]])
    k = kernel_rbf(A, B, 0.5)
    assert np.shape(k) == (3, 2), f"com 3 linhas em A e 2 em B, veio o formato {np.shape(k)}; são 3 × 2"
    assert perto(k[2, 1], math.exp(-0.5 * 4), 1e-12), "a posição [2, 1] é o pedido 3 de A com o pedido 2 de B"


def testar_par_da_aula():
    a = MARCO.loc[[fr.PAR[0]], fr.MAPA].to_numpy(float)
    b = MARCO.loc[[fr.PAR[1]], fr.MAPA].to_numpy(float)
    k = kernel_rbf(a, b, GAMMA)[0, 0]
    assert perto(k, 0.0790, 0.00005), f"o par {fr.PAR[0]} e {fr.PAR[1]} deu {k:.4f}; a parte 6 da aula tem 0,0790"


def testar_gamma_invalido():
    for g in (0.0, -1.0):
        try:
            kernel_rbf(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), g)
        except ValueError:
            continue
        raise AssertionError(f"kernel_rbf aceitou gamma = {g}; tinha que parar com ValueError")


def testar_colunas_diferentes():
    try:
        kernel_rbf(np.array([[1.0]]), np.array([[1.0, 2.0]]), 0.5)
    except ValueError:
        return
    raise AssertionError("kernel_rbf aceitou A com 1 coluna e B com 2; tinha que parar com ValueError")


def testar_nao_muda_a_entrada():
    A, B = XM[:5].copy(), XM[5:9].copy()
    kernel_rbf(A, B, 0.1)
    assert np.array_equal(A, XM[:5]) and np.array_equal(B, XM[5:9]), "kernel_rbf mudou o que recebeu"


def testar_decisao_da_aula():
    f = decisao(RETA, X10)
    esperado = [1.00, 1.80, 1.96, 1.88, 1.08, -1.00, -1.16, -1.40, -1.32, -1.48]
    for pid, a, b in zip(fr.DEZ, f, esperado):
        assert perto(a, b), f"o pedido {pid} deu f = {a:.2f}; a parte 1 da aula tem {b:.2f}"


def testar_decisao_usa_o_b():
    m = dict(RETA, alfa=np.zeros(len(Y10)), b=0.7)
    f = decisao(m, X10[:3])
    assert all(perto(x, 0.7, 1e-12) for x in f), f"com todos os alfas em zero, f tinha que ser o b, 0,7; deu {f}"


def testar_decisao_usa_o_y():
    m = {"X": np.array([[1.0, 0.0]]), "y": np.array([-1.0]), "alfa": np.array([2.0]), "b": 0.5,
         "kernel": fr.k_linear, "C": 10.0}
    f = decisao(m, np.array([[3.0, 0.0]]))[0]
    assert perto(f, 2 * -1 * 3 + 0.5, 1e-12), f"um pedido de y = −1, alfa 2 e kernel 3 dá 2 × (−1) × 3 + 0,5 = −5,5; deu {f}"


def testar_decisao_usa_o_kernel_do_modelo():
    m = fr.treinar(XM, YM, lambda A, B: fr.k_rbf(A, B, GAMMA), 1.0)
    Xa = ABRIL[fr.MAPA].to_numpy(float)
    meu, aula = decisao(m, Xa), fr.decisao(m, Xa)
    assert np.shape(meu) == np.shape(aula), f"para os {len(Xa)} pedidos de abril, vieram {np.shape(meu)} valores"
    assert np.abs(meu - aula).max() < 1e-9, "com o modelo RBF, decisao deu outro f; use modelo['kernel']"


def testar_decisao_colunas():
    try:
        decisao(RETA, np.array([[9.0]]))
    except ValueError:
        return
    raise AssertionError("decisao aceitou um pedido com 1 coluna num modelo de 2; tinha que parar com ValueError")


def testar_escrita_no_proprio_arquivo():
    for f in (kernel_rbf, decisao):
        assert f.__module__ == "meu_kernel", f"{f.__name__} veio de {f.__module__}; escreva no meu_kernel.py"
    assert meu_kernel.decisao is decisao


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
