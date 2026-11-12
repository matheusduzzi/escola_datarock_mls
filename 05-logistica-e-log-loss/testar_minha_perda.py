"""Testes das funções sigmoide, log_loss e gradiente do minha_perda.py.

Rode de dentro da pasta da aula:  python3 testar_minha_perda.py
Os testes usam os quatro pedidos da aula e o chute -3 + 0,2 × prazo, não os pedidos da Bia:
eles mostram que o código faz a conta da aula, e a conta do exercício continua com você.
"""
import math

import numpy as np

from minha_perda import gradiente, log_loss, sigmoide

PRAZO = [5, 10, 15, 15]
CANCELOU = [0, 0, 1, 0]


def perto(a, b, tol=1e-6):
    return abs(a - b) < tol


def para_com_value_error(funcao, *args):
    try:
        funcao(*args)
    except ValueError:
        return True
    return False


def testar_sigmoide_no_zero_e_simetria():
    assert perto(float(sigmoide(0)), 0.5), f"sigmoide(0) deu {sigmoide(0)}, a aula tem 0,5"
    assert perto(float(sigmoide(-1)), 0.268941), f"sigmoide(-1) deu {sigmoide(-1)}, a aula tem 0,2689"
    assert perto(float(sigmoide(2) + sigmoide(-2)), 1.0), "sigmoide(2) + sigmoide(-2) tinha que dar 1"


def testar_sigmoide_aceita_lista():
    notas = np.asarray(sigmoide([-2, 0, 2]), dtype=float)
    assert notas.shape == (3,), f"com uma lista de 3 scores saíram {notas.shape}"
    assert np.all(np.diff(notas) > 0), "score maior tinha que dar nota maior"


def testar_log_loss_da_aula():
    notas = sigmoide(-3 + 0.2 * np.asarray(PRAZO, dtype=float))
    valor = log_loss(CANCELOU, notas)
    assert perto(valor, 0.456621), f"log-loss do chute deu {valor}, a aula tem 0,4566"


def testar_log_loss_um_pedido():
    assert perto(log_loss([1], [0.5]), math.log(2)), "cancelou com nota 0,5: a perda é ln 2 = 0,6931"
    assert perto(log_loss([0], [0.9]), math.log(10)), "entregue com nota 0,9: a perda é ln 10 = 2,3026"


def testar_log_loss_nota_extrema():
    valor = log_loss([1, 0], [0.0, 1.0])
    assert math.isfinite(valor) and valor > 30, f"nota 0 para um cancelado deu {valor}; tinha que ser grande e finita"


def testar_log_loss_rotulo_invalido():
    for y in ([0, 2], [0.5, 1]):
        assert para_com_value_error(log_loss, y, [0.3, 0.3]), f"log_loss aceitou o rótulo {y}; só vale 0 ou 1"


def testar_log_loss_nota_fora():
    for p in ([1.2, 0.3], [-0.1, 0.3]):
        assert para_com_value_error(log_loss, [1, 0], p), f"log_loss aceitou a nota {p}; nota é de 0 a 1"


def testar_log_loss_tamanhos():
    assert para_com_value_error(log_loss, [1, 0, 0], [0.3, 0.3]), "log_loss aceitou 3 rótulos e 2 notas"
    assert para_com_value_error(log_loss, [1, 0, 0], [0.3]), "log_loss aceitou 3 rótulos e 1 nota só"


def testar_gradiente_da_aula():
    d_a, d_b = gradiente(-3, 0.2, PRAZO, CANCELOU)
    assert perto(d_a, 0.097037, 1e-5), f"derivada no a deu {d_a}, a aula tem 0,0970"
    assert perto(d_b, 0.821355, 1e-5), f"derivada no b deu {d_b}, a aula tem 0,8214"


def testar_gradiente_bate_com_a_variacao():
    # A derivada tem que bater com "mexer um pouco e ver quanto mudou", em outro ponto.
    a, b, h = -1.0, 0.05, 1e-6
    x = np.asarray(PRAZO, dtype=float)
    y = np.asarray(CANCELOU, dtype=float)

    def perda(a, b):
        p = 1 / (1 + np.exp(-(a + b * x)))
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    d_a, d_b = gradiente(a, b, PRAZO, CANCELOU)
    assert abs(d_a - (perda(a + h, b) - perda(a - h, b)) / (2 * h)) < 1e-5, "derivada no a não bate com a variação"
    assert abs(d_b - (perda(a, b + h) - perda(a, b - h)) / (2 * h)) < 1e-5, "derivada no b não bate com a variação"


def testar_gradiente_tamanhos():
    assert para_com_value_error(gradiente, -3, 0.2, [5, 10, 15, 15], [0]), "gradiente aceitou 4 prazos e 1 rótulo"


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
