"""Tarefa 2 do diagnóstico: prever com uma reta e medir o erro.

Escreva as duas funções com o NumPy, sem loop em Python. Depois rode:

    python3 tarefa2.py
    python3 testar_tarefa2.py
"""
import numpy as np


def prever(a, b, x):
    """Retorna a + b * x para cada valor de x, como array do NumPy.

    x pode chegar como lista ou como array.
    """
    raise NotImplementedError("escreva a função prever")


def erro_quadratico_medio(y, previsto):
    """Retorna a média de (y - previsto) ao quadrado.

    Para com ValueError se y e previsto não tiverem o mesmo tamanho.
    """
    raise NotImplementedError("escreva a função erro_quadratico_medio")


if __name__ == "__main__":
    itens = [1, 2, 4, 5]
    valor = [70, 85, 150, 160]
    previsto = prever(40, 25, itens)
    print("previsto:", previsto)
    print("erro quadrático médio:", erro_quadratico_medio(valor, previsto))
