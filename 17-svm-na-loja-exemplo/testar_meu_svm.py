"""Testes das funções padronizar e no_teto do meu_svm.py.

Rode de dentro da pasta da aula:  python3 testar_meu_svm.py
Os testes usam casos pequenos feitos à mão e os números da parte 2 da aula. A leitura das partes
A e B do meu_svm.py continua com você.
"""
import math

import numpy as np
import pandas as pd

import meu_svm
from meu_svm import no_teto, padronizar
from svm_loja import NUMERICAS, dobras, ler


def _erro(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


# ---------- padronizar ----------

def testar_media_e_desvio_do_treino():
    tr = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    ou = pd.DataFrame({"a": [4.0]})
    ptr, pou = padronizar(tr, ou)
    d = math.sqrt(2 / 3)
    assert np.allclose(ptr["a"], [-1 / d, 0, 1 / d]), f"treino [1, 2, 3]: média 2, desvio √(2/3); veio {list(ptr['a'])}"
    assert np.isclose(pou["a"].iloc[0], 2 / d), f"outro [4] com a média e o desvio do treino: (4 − 2) ÷ √(2/3); veio {pou['a'].iloc[0]}"


def testar_desvio_da_populacao():
    ptr, _ = padronizar(pd.DataFrame({"a": [0.0, 2.0]}), pd.DataFrame({"a": [1.0]}))
    assert np.allclose(ptr["a"], [-1, 1]), f"[0, 2] vira [−1, 1] com o desvio da população (ddof=0); veio {list(ptr['a'])}"


def testar_outro_nao_entra_na_conta():
    tr = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    ptr1, _ = padronizar(tr, pd.DataFrame({"a": [4.0]}))
    ptr2, _ = padronizar(tr, pd.DataFrame({"a": [4000.0, -50.0]}))
    assert np.allclose(ptr1["a"], ptr2["a"]), "o treino padronizado não pode mudar quando o outro muda"


def testar_cada_coluna_com_a_sua_conta():
    tr = pd.DataFrame({"a": [0.0, 2.0], "b": [10.0, 30.0]})
    ptr, pou = padronizar(tr, pd.DataFrame({"a": [4.0], "b": [20.0]}))
    assert np.allclose(ptr["b"], [-1, 1]) and np.isclose(pou["a"].iloc[0], 3) and np.isclose(pou["b"].iloc[0], 0), \
        f"a: média 1, desvio 1; b: média 20, desvio 10. Veio treino {ptr.to_dict('list')}, outro {pou.to_dict('list')}"


def testar_unidade_nao_muda_o_resultado():
    tr = pd.DataFrame({"valor": [870.33, 214.95, 99.9]})
    ou = pd.DataFrame({"valor": [300.0]})
    a, b = padronizar(tr, ou)
    c, d = padronizar(tr * 100, ou * 100)
    assert np.allclose(a, c) and np.allclose(b, d), "o valor em centavos precisa dar o mesmo resultado padronizado"


def testar_mesmo_indice_e_colunas():
    tr = pd.DataFrame({"a": [1.0, 3.0], "b": [5.0, 7.0]}, index=[1978, 1545])
    ou = pd.DataFrame({"a": [2.0], "b": [6.0]}, index=[22])
    ptr, pou = padronizar(tr, ou)
    assert isinstance(ptr, pd.DataFrame) and isinstance(pou, pd.DataFrame), "retorne dois DataFrames"
    assert list(ptr.index) == [1978, 1545] and list(pou.index) == [22], "mantenha o índice (o número do pedido)"
    assert list(ptr.columns) == ["a", "b"] and list(pou.columns) == ["a", "b"], "mantenha as colunas, na mesma ordem"


def testar_colunas_diferentes():
    tr = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 3.0]})
    assert _erro(padronizar, tr, pd.DataFrame({"a": [1.0]})), "outro sem a coluna b: precisa parar com ValueError"
    assert _erro(padronizar, tr, pd.DataFrame({"b": [1.0], "a": [1.0]})), \
        "as mesmas colunas em outra ordem: precisa parar com ValueError"


def testar_valor_faltando():
    tr = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
    assert _erro(padronizar, tr, pd.DataFrame({"a": [1.0]})), "valor faltando no treino: precisa parar com ValueError"
    assert _erro(padronizar, pd.DataFrame({"a": [1.0, 3.0]}), pd.DataFrame({"a": [float("nan")]})), \
        "valor faltando no outro: precisa parar com ValueError"


def testar_desvio_zero():
    tr = pd.DataFrame({"a": [1.0, 2.0], "b": [5.0, 5.0]})
    assert _erro(padronizar, tr, pd.DataFrame({"a": [1.0], "b": [5.0]})), \
        "a coluna b tem desvio zero no treino: dividir por zero não pode passar calado"


def testar_padronizar_nao_muda_o_que_recebeu():
    tr = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    ou = pd.DataFrame({"a": [4.0]})
    padronizar(tr, ou)
    assert list(tr["a"]) == [1.0, 2.0, 3.0] and list(ou["a"]) == [4.0], "padronizar mudou o DataFrame que recebeu"


def testar_media_e_desvio_da_aula():
    _, tr, me = dobras(ler())[2]
    ptr, pme = padronizar(tr[NUMERICAS].astype(float), me[NUMERICAS].astype(float))
    assert np.allclose(ptr.mean(), 0) and np.allclose(ptr.std(ddof=0), 1), "o treino padronizado tem média 0 e desvio 1"
    linha = pme[me["id_pedido"].to_numpy() == 1978].iloc[0]
    assert round(linha["valor"], 2) == round((870.33 - 346.70) / 380.15, 2), \
        f"pedido 1978: (870,33 − 346,70) ÷ 380,15, como na parte 2 da aula; veio {linha['valor']:.4f}"


# ---------- no_teto ----------

def testar_teto_simples():
    f = pd.Series([3.0, 1.0, 2.0], index=[10, 11, 12])
    y = pd.Series([1, 0, 0], index=[10, 11, 12])
    assert no_teto(f, y, 1) == 1, "o pedido 10 tem o maior f e cancelou: 1"
    assert no_teto(f, y, 2) == 1, "os dois maiores f são os pedidos 10 e 12; só o 10 cancelou: 1"
    assert no_teto(f, y, 3) == 1, "com os três: 1"


def testar_f_negativo():
    f = pd.Series([-1.0005, -0.9999, -1.0001], index=[1, 2, 3])
    y = pd.Series([0, 1, 0], index=[1, 2, 3])
    assert no_teto(f, y, 1) == 1, "o maior f é −0,9999, do pedido 2, que cancelou"


def testar_empate_pelo_numero_do_pedido():
    f = pd.Series([0.5, 0.5, 0.5], index=[10, 2, 5])
    y = pd.Series([0, 1, 0], index=[10, 2, 5])
    assert no_teto(f, y, 1) == 1, "f empatado: vem antes o pedido de número menor, o 2, que cancelou"
    y2 = pd.Series([1, 0, 0], index=[10, 2, 5])
    assert no_teto(f, y2, 2) == 0, "f empatado: os dois primeiros são os pedidos 2 e 5, nenhum cancelou"


def testar_indices_diferentes():
    f = pd.Series([1.0, 2.0], index=[1, 2])
    assert _erro(no_teto, f, pd.Series([0, 1], index=[1, 3]), 1), "índices diferentes: precisa parar com ValueError"
    assert _erro(no_teto, f, pd.Series([1, 0], index=[2, 1]), 1), \
        "o mesmo índice em outra ordem: precisa parar com ValueError, para não juntar f e y de pedidos trocados"


def testar_quantos_fora_da_faixa():
    f = pd.Series([1.0, 2.0], index=[1, 2])
    y = pd.Series([0, 1], index=[1, 2])
    assert _erro(no_teto, f, y, 0), "quantos = 0: precisa parar com ValueError"
    assert _erro(no_teto, f, y, 3), "quantos maior que o número de pedidos: precisa parar com ValueError"


def testar_f_faltando():
    f = pd.Series([1.0, float("nan")], index=[1, 2])
    assert _erro(no_teto, f, pd.Series([0, 1], index=[1, 2]), 1), "f faltando: precisa parar com ValueError"


def testar_y_so_zero_e_um():
    f = pd.Series([1.0, 2.0], index=[1, 2])
    assert _erro(no_teto, f, pd.Series([0, 2], index=[1, 2]), 1), "y com 2: precisa parar com ValueError"


def testar_no_teto_nao_muda_o_que_recebeu():
    f = pd.Series([1.0, 3.0, 2.0], index=[7, 8, 9])
    y = pd.Series([0, 1, 1], index=[7, 8, 9])
    no_teto(f, y, 2)
    assert list(f) == [1.0, 3.0, 2.0] and list(f.index) == [7, 8, 9] and list(y) == [0, 1, 1], \
        "no_teto mudou a Series que recebeu"


def testar_escrita_no_proprio_arquivo():
    for fn in (padronizar, no_teto):
        assert fn.__module__ == "meu_svm", f"{fn.__name__} veio de {fn.__module__}; escreva no meu_svm.py"
    assert meu_svm.padronizar is padronizar and meu_svm.no_teto is no_teto


if __name__ == "__main__":
    import sys

    testes = [(nome, fn) for nome, fn in sorted(globals().items()) if nome.startswith("testar_")]
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
