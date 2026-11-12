"""Testes das funções mse e melhor_reta do minha_reta.py.

Rode de dentro da pasta da aula:  python3 testar_minha_reta.py
Os testes usam os quatro pedidos da aula, não os pedidos da Bia: eles mostram que o código
faz a conta da aula, e a conta do exercício continua com você.
"""
import numpy as np

from minha_reta import mse, melhor_reta

ITENS = [1, 2, 4, 5]
VALOR = [70, 85, 150, 160]


def perto(a, b):
    return abs(a - b) < 1e-9


def erro_da_reta(a, b):
    # Conta feita aqui mesmo, sem a sua função, para comparar com ela.
    return float(np.mean((np.array(VALOR) - (a + b * np.array(ITENS))) ** 2))


def testar_mse_das_duas_retas_da_aula():
    e1, e2 = mse(40, 25, ITENS, VALOR), mse(30, 27, ITENS, VALOR)
    assert perto(e1, 43.75), f"MSE da reta 40 + 25 × itens deu {e1}, a aula tem 43,75"
    assert perto(e2, 84.75), f"MSE da reta 30 + 27 × itens deu {e2}, a aula tem 84,75"


def testar_mse_aceita_array():
    e = mse(40.0, 25.0, np.array(ITENS, dtype=float), np.array(VALOR, dtype=float))
    assert perto(e, 43.75), f"com array do NumPy o MSE deu {e}"


def testar_mse_tamanhos_diferentes():
    try:
        mse(40, 25, [1, 2, 4, 5], [70])
    except ValueError:
        return
    raise AssertionError("mse aceitou 4 itens e 1 valor; tinha que parar com ValueError")


def testar_melhor_reta_da_aula():
    a, b = melhor_reta(ITENS, VALOR)
    assert perto(a, 42.75) and perto(b, 24.5), f"melhor_reta deu ({a}; {b}), a aula tem a = 42,75 e b = 24,5"


def testar_melhor_reta_ganha_das_vizinhas():
    a, b = melhor_reta(ITENS, VALOR)
    melhor = erro_da_reta(a, b)
    for da, db in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1), (0.1, -0.1), (-0.1, 0.1)]:
        vizinha = erro_da_reta(a + da, b + db)
        assert melhor < vizinha, f"a reta ({a + da}; {b + db}) erra menos que a sua melhor reta"


def testar_residuos_da_melhor_reta():
    a, b = melhor_reta(ITENS, VALOR)
    residuo = np.array(VALOR) - (a + b * np.array(ITENS))
    assert abs(residuo.sum()) < 1e-9, f"na melhor reta os resíduos somam {residuo.sum()}, e não zero"
    assert abs((residuo * np.array(ITENS)).sum()) < 1e-9, "na melhor reta a soma de resíduo × itens não deu zero"


def testar_melhor_reta_tamanhos_diferentes():
    try:
        melhor_reta([1, 2, 4, 5], [70])
    except ValueError:
        return
    raise AssertionError("melhor_reta aceitou 4 itens e 1 valor; tinha que parar com ValueError")


def testar_itens_todos_iguais():
    try:
        melhor_reta([3, 3, 3], [90, 110, 130])
    except ValueError:
        return
    raise AssertionError("melhor_reta aceitou três pedidos com 3 itens; não existe uma inclinação só, "
                         "tinha que parar com ValueError")


if __name__ == "__main__":
    import sys
    import warnings

    warnings.simplefilter("ignore")
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
