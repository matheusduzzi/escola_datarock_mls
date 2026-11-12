"""Testes das funções dividir_em_dobras e erro_nas_dobras do minha_escolha.py.

Rode de dentro da pasta da aula:  python3 testar_minha_escolha.py
Os testes usam seis pedidos inventados aqui, não os pedidos da loja: eles mostram que o código
faz a conta certa, e a escolha do alfa continua com você.
"""
import sys
import traceback

import numpy as np
import pandas as pd

from minha_escolha import dividir_em_dobras, erro_nas_dobras

X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6.0], "b": [0, 1, 0, 1, 1, 0.0]})
Y = np.array([2, 4, 5, 4, 6, 9.0])
DOBRAS = np.array([0, 1, 2, 0, 1, 2])


def perto(a, b):
    return all(abs(u - v) < 1e-6 for u, v in zip(a, b)) and len(a) == len(b)


def falha_com_valueerror(funcao, *args):
    try:
        funcao(*args)
    except ValueError:
        return True
    return False


def testar_tamanho_das_dobras():
    d = np.asarray(dividir_em_dobras(12, 5, 3))
    assert len(d) == 12, f"vieram {len(d)} dobras para 12 pedidos"
    assert np.bincount(d).tolist() == [3, 3, 2, 2, 2], f"pedidos por dobra {np.bincount(d).tolist()}, eram 3, 3, 2, 2, 2"


def testar_regra_do_sorteio():
    d = np.asarray(dividir_em_dobras(7, 3, 14)).tolist()
    assert d == [2, 1, 2, 1, 0, 0, 0], f"com 7 pedidos, 3 dobras e o sorteio 14 veio {d}; a regra da aula dá [2, 1, 2, 1, 0, 0, 0]"


def testar_mesmo_sorteio_mesmas_dobras():
    a, b = np.asarray(dividir_em_dobras(12, 5, 3)), np.asarray(dividir_em_dobras(12, 5, 3))
    assert a.tolist() == b.tolist(), "o mesmo sorteio deu dobras diferentes"
    c = np.asarray(dividir_em_dobras(12, 5, 4))
    assert a.tolist() != c.tolist(), "os sorteios 3 e 4 deram as mesmas dobras; o sorteio não está sendo usado"


def testar_numero_de_dobras_invalido():
    assert falha_com_valueerror(dividir_em_dobras, 7, 1, 14), "aceitou 1 dobra; tinha que parar com ValueError"
    assert falha_com_valueerror(dividir_em_dobras, 7, 8, 14), "aceitou 8 dobras para 7 pedidos; tinha que parar com ValueError"


def testar_erro_sem_penalidade():
    e = erro_nas_dobras(X, Y, DOBRAS, 0.0, 0.0)
    assert perto(e, [2.25, 4.073753462603878, 10.306122448979593]), f"MSE por dobra sem penalidade: {e}"


def testar_padroniza_so_com_as_outras_dobras():
    e = erro_nas_dobras(X, Y, DOBRAS, 0.0, 1.0)
    assert perto(e, [5.339414858645628, 0.4459209157127991, 12.804733727810651]), (
        f"MSE por dobra com Ridge 1: {e}. Se deu [7,00; 0,12; 11,91], a média e o desvio vieram de todas as linhas")


def testar_lasso():
    e = erro_nas_dobras(X, Y, DOBRAS, 2.0, 0.0)
    assert perto(e, [5.373156617270721, 0.01676718544497522, 10.613153149862672]), f"MSE por dobra com Lasso 2: {e}"


def testar_tamanhos_diferentes():
    assert falha_com_valueerror(erro_nas_dobras, X, Y, DOBRAS[:5], 0.0, 1.0), (
        "aceitou 6 linhas e 5 dobras; tinha que parar com ValueError")
    assert falha_com_valueerror(erro_nas_dobras, X, Y[:5], DOBRAS, 0.0, 1.0), (
        "aceitou 6 linhas e 5 valores; tinha que parar com ValueError")


def testar_dobra_vazia():
    assert falha_com_valueerror(erro_nas_dobras, X, Y, np.array([0, 0, 2, 2, 0, 2]), 0.0, 1.0), (
        "aceitou as dobras 0 e 2 sem a dobra 1; tinha que parar com ValueError")


def testar_nao_muda_os_dados():
    x = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6.0], "b": [0, 1, 0, 1, 1, 0.0]})
    copia = x.copy()
    erro_nas_dobras(x, Y, DOBRAS, 0.0, 1.0)
    assert x.equals(copia), "erro_nas_dobras mudou o DataFrame que recebeu"


TESTES = [testar_tamanho_das_dobras, testar_regra_do_sorteio, testar_mesmo_sorteio_mesmas_dobras,
          testar_numero_de_dobras_invalido, testar_erro_sem_penalidade, testar_padroniza_so_com_as_outras_dobras,
          testar_lasso, testar_tamanhos_diferentes, testar_dobra_vazia, testar_nao_muda_os_dados]


def main():
    falhas = 0
    for teste in TESTES:
        try:
            teste()
        except NotImplementedError as erro:
            print(f"{teste.__name__}: NotImplementedError: {erro}")
            falhas += 1
        except AssertionError as erro:
            print(f"{teste.__name__}: falhou: {erro}")
            falhas += 1
        except Exception:
            print(f"{teste.__name__}: erro inesperado:")
            traceback.print_exc(limit=1)
            falhas += 1
    if falhas:
        print(f"{falhas} de {len(TESTES)} testes falharam")
        return 1
    print(f"ok: {len(TESTES)} testes passaram")
    return 0


if __name__ == "__main__":
    sys.exit(main())
