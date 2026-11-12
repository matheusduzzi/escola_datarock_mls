"""Teste da tarefa 2. Rode de dentro desta pasta:

    python3 testar_tarefa2.py

Cada teste que passa escreve uma linha com "ok". O primeiro que falha para o programa.
"""
import numpy as np

from tarefa2 import erro_quadratico_medio, prever


def testa_prever_com_lista():
    p = prever(30, 28, [1, 3, 6])
    assert isinstance(p, np.ndarray), "prever precisa retornar um array do NumPy"
    assert np.allclose(p, [58, 114, 198]), f"previsto errado: {p}"
    print("ok prever com lista")


def testa_erro_quadratico_medio():
    e = erro_quadratico_medio(np.array([60, 120, 190]), prever(30, 28, [1, 3, 6]))
    assert abs(e - 104 / 3) < 1e-9, f"erro quadrático médio errado: {e}"
    print("ok erro quadrático médio")


def testa_erro_zero():
    assert erro_quadratico_medio([10, 20], [10, 20]) == 0
    print("ok erro zero quando a previsão acerta tudo")


def testa_tamanhos_diferentes():
    for y, p in (([1, 2, 3, 4], [1, 2, 3]), ([1, 2, 3, 4], [2])):
        try:
            erro_quadratico_medio(y, p)
        except ValueError:
            continue
        raise AssertionError(f"tamanhos {len(y)} e {len(p)} deviam parar com ValueError")
    print("ok tamanhos diferentes param com ValueError")


if __name__ == "__main__":
    testa_prever_com_lista()
    testa_erro_quadratico_medio()
    testa_erro_zero()
    testa_tamanhos_diferentes()
    print("todos os testes passaram")
