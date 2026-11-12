"""Testes das funções distancias e prever do meus_vizinhos.py.

Rode de dentro da pasta da aula:  python3 testar_meus_vizinhos.py
Os testes usam a compra 6912 e as seis compras da parte 1 da aula, não as de junho: eles
mostram que o código faz a conta da aula, e a leitura de junho continua com você.
"""
import numpy as np
import pandas as pd

import meus_vizinhos
from meus_vizinhos import distancias, prever
from vizinhos import CANDIDATAS, COLUNAS, MAO, NOVA, carregar, separar

DADOS = carregar()
CONH, _ = separar(DADOS)
CAND = DADOS.loc[CANDIDATAS, MAO]
ROT = DADOS.loc[CANDIDATAS, "fraude"].to_numpy()
NOVA_MAO = DADOS.loc[NOVA, MAO]
DESVIO = CONH[MAO].std(ddof=0)
UM = pd.Series(1.0, index=MAO)


def perto(a, b, tol=0.005):
    return abs(a - b) < tol


def testar_distancias_padronizadas_da_aula():
    d = distancias(CAND, NOVA_MAO, DESVIO)
    esperado = [5.17, 5.17, 5.17, 0.25, 1.32, 1.33]
    assert len(d) == 6, f"vieram {len(d)} distâncias; são seis compras"
    for i, (a, b) in enumerate(zip(d, esperado)):
        assert perto(a, b), f"a compra {CANDIDATAS[i]} deu {a:.2f}; a parte 1 da aula tem {b:.2f}"


def testar_distancias_sem_padronizar_da_aula():
    d = distancias(CAND, NOVA_MAO, UM)
    esperado = [15.07, 25.51, 31.19, 86.65, 88.39, 113.64]
    for i, (a, b) in enumerate(zip(d, esperado)):
        assert perto(a, b), f"com desvio 1, a compra {CANDIDATAS[i]} deu {a:.2f}; a aula tem {b:.2f}"


def testar_pitagoras():
    conh = pd.DataFrame({"a": [3.0], "b": [4.0]})
    d = distancias(conh, pd.Series({"a": 0.0, "b": 0.0}), pd.Series({"a": 1.0, "b": 1.0}))
    assert perto(d[0], 5, 1e-9), f"de (0, 0) até (3, 4) deu {d[0]}; por Pitágoras, a raiz de 9 + 16 é 5"


def testar_divide_pelo_desvio():
    conh = pd.DataFrame({"a": [10.0], "b": [0.0]})
    d = distancias(conh, pd.Series({"a": 0.0, "b": 0.0}), pd.Series({"a": 5.0, "b": 1.0}))
    assert perto(d[0], 2, 1e-9), f"diferença 10 com desvio 5 deu {d[0]}; são 10 ÷ 5 = 2 desvios"


def testar_voto_da_aula():
    assert prever(distancias(CAND, NOVA_MAO, DESVIO), ROT, 3) == (1, 2), (
        "padronizada, K = 3 tinha que dar fraude com 2 de 3 votos (parte 1 da aula)")
    assert prever(distancias(CAND, NOVA_MAO, UM), ROT, 3) == (0, 0), (
        "sem padronizar, K = 3 tinha que dar legítima com 0 de 3 votos (parte 1 da aula)")


def testar_empate_fica_com_a_primeira():
    dist = np.array([2.0, 1.0, 1.0, 1.0])
    rot = np.array([1, 0, 1, 1])
    assert prever(dist, rot, 1) == (0, 0), "no empate, o vizinho é a primeira compra das que empatam (a de rótulo 0)"
    assert prever(dist, rot, 3) == (1, 2), "com K = 3, ficam as três compras de distância 1"


def testar_k_invalido():
    dist, rot = np.array([1.0, 2.0, 3.0]), np.array([0, 1, 1])
    for k in (0, 2, 4, -1, 1.0):
        try:
            prever(dist, rot, k)
        except ValueError:
            continue
        raise AssertionError(f"prever aceitou k = {k!r}; k precisa ser ímpar, de 1 até 3")


def testar_colunas_diferentes():
    try:
        distancias(CAND, NOVA_MAO[["tentativas_cartao", "valor"]], DESVIO)
    except ValueError:
        return
    raise AssertionError("distancias aceitou a compra nova com as colunas em outra ordem; tinha que parar com ValueError")


def testar_desvio_zero():
    try:
        distancias(CAND, NOVA_MAO, DESVIO.where(DESVIO.index != "valor", 0.0))
    except ValueError:
        return
    raise AssertionError("distancias aceitou desvio zero; tinha que parar com ValueError")


def testar_nao_muda_a_entrada():
    c, n, d = CAND.copy(), NOVA_MAO.copy(), DESVIO.copy()
    distancias(CAND, NOVA_MAO, DESVIO)
    assert CAND.equals(c) and NOVA_MAO.equals(n) and DESVIO.equals(d), "distancias mudou o que recebeu"


def testar_cinco_colunas_da_aula():
    desvio = CONH[COLUNAS].std(ddof=0)
    previsao, votos = prever(distancias(CONH[COLUNAS], DADOS.loc[NOVA, COLUNAS], desvio),
                             CONH["fraude"].to_numpy(), 5)
    assert (previsao, votos) == (1, 4), f"a compra {NOVA} com as cinco colunas deu {votos} de 5; a parte 2 da aula tem 4"


def testar_escrita_no_proprio_arquivo():
    for f in (distancias, prever):
        assert f.__module__ == "meus_vizinhos", f"{f.__name__} veio de {f.__module__}; escreva no meus_vizinhos.py"
    assert meus_vizinhos.distancias is distancias


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
