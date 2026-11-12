"""Testes das funções gradiente e passo do minha_atualizacao.py.

Rode de dentro da pasta da aula:  python3 testar_minha_atualizacao.py
Os testes usam os quatro pedidos da aula, não os pedidos da Rê: eles mostram que o código
faz a conta da aula, e a conta do exercício continua com você.
"""
import numpy as np

from minha_atualizacao import gradiente, passo

ITENS = [1, 2, 4, 5]
VALOR = [70, 85, 150, 160]


def perto(a, b):
    return abs(a - b) < 1e-9


def testar_gradiente_na_reta_da_aula():
    d_a, d_b = gradiente(40, 25, ITENS, VALOR)
    assert perto(d_a, -2.5), f"derivada no a deu {d_a}, a aula tem -2,5"
    assert perto(d_b, -5.0), f"derivada no b deu {d_b}, a aula tem -5"


def testar_gradiente_aceita_array():
    d_a, d_b = gradiente(40.0, 25.0, np.array(ITENS, dtype=float), np.array(VALOR, dtype=float))
    assert perto(d_a, -2.5) and perto(d_b, -5.0), "com array do NumPy o resultado mudou"


def testar_gradiente_zero_na_melhor_reta():
    d_a, d_b = gradiente(42.75, 24.5, ITENS, VALOR)
    assert perto(d_a, 0) and perto(d_b, 0), f"na reta de menor MSE o gradiente deu ({d_a}; {d_b})"


def testar_gradiente_bate_com_a_variacao():
    # A derivada tem que bater com "mexer um pouco e ver quanto mudou", em qualquer ponto.
    a, b, h = 30.0, 27.0, 1e-6
    mse = lambda a, b: np.mean((np.array(VALOR) - (a + b * np.array(ITENS))) ** 2)
    d_a, d_b = gradiente(a, b, ITENS, VALOR)
    assert abs(d_a - (mse(a + h, b) - mse(a - h, b)) / (2 * h)) < 1e-4, "derivada no a não bate com a variação"
    assert abs(d_b - (mse(a, b + h) - mse(a, b - h)) / (2 * h)) < 1e-4, "derivada no b não bate com a variação"


def testar_passo_da_aula():
    a1, b1 = passo(40, 25, ITENS, VALOR, 0.01)
    assert perto(a1, 40.025) and perto(b1, 25.05), f"o passo deu ({a1}; {b1}), a aula tem 40,025 e 25,05"


def testar_passo_desce_o_erro():
    mse = lambda a, b: np.mean((np.array(VALOR) - (a + b * np.array(ITENS))) ** 2)
    a1, b1 = passo(40, 25, ITENS, VALOR, 0.01)
    assert mse(a1, b1) < mse(40, 25), "com taxa 0,01 o MSE tinha que cair"


def testar_tamanhos_diferentes():
    try:
        gradiente(40, 25, [1, 2, 4, 5], [70])
    except ValueError:
        return
    raise AssertionError("gradiente aceitou 4 itens e 1 valor; tinha que parar com ValueError")


def testar_taxa_invalida():
    for taxa in (0, -0.01):
        try:
            passo(40, 25, ITENS, VALOR, taxa)
        except ValueError:
            continue
        raise AssertionError(f"passo aceitou taxa {taxa}; tinha que parar com ValueError")


if __name__ == "__main__":
    import sys

    testes = [(nome, f) for nome, f in sorted(globals().items()) if nome.startswith("testar_")]
    falhas = 0
    for nome, teste in testes:
        try:
            teste()
        except AssertionError as erro:
            falhas += 1
            print(f"FALHOU {nome}: {erro}")
    if falhas:
        print(f"{falhas} de {len(testes)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(testes)} testes passaram")
