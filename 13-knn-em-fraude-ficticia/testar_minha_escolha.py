"""Testes das funções dobras_estratificadas e custo_da_regra do minha_escolha.py.

Rode de dentro da pasta da aula:  python3 testar_minha_escolha.py
Os testes usam compras inventadas aqui dentro e as dobras e os custos da aula, com a análise a
R$ 15. Eles não usam o preço do Diego: a escolha com R$ 40 continua com você.
"""
import hashlib

import numpy as np

import benchmark
import minha_escolha
from minha_escolha import custo_da_regra, dobras_estratificadas

PEQUENO = np.array([0] * 20 + [1] * 7)


def testar_dobras_tamanho_e_valores():
    d = np.asarray(dobras_estratificadas(PEQUENO, 3, 0))
    assert len(d) == len(PEQUENO), f"vieram {len(d)} dobras para {len(PEQUENO)} compras"
    assert sorted(set(d.tolist())) == [0, 1, 2], f"as dobras precisam ser 0, 1 e 2; vieram {sorted(set(d.tolist()))}"


def testar_fraudes_repartidas():
    rotulos = np.array([0] * 90 + [1] * 10)
    for sorteio in range(10):
        d = np.asarray(dobras_estratificadas(rotulos, 5, sorteio))
        fraudes = [int(rotulos[d == j].sum()) for j in range(5)]
        legitimas = [int((rotulos[d == j] == 0).sum()) for j in range(5)]
        assert fraudes == [2] * 5, f"no sorteio {sorteio}, 10 fraudes em 5 dobras saíram {fraudes}; o certo é 2 em cada"
        assert legitimas == [18] * 5, f"no sorteio {sorteio}, 90 legítimas em 5 dobras saíram {legitimas}; o certo é 18"


def testar_dobras_da_aula():
    treino = benchmark.carregar_treino()
    d = np.asarray(dobras_estratificadas(treino["fraude"].to_numpy(), 5, 13))
    ids = treino["id_transacao"].to_numpy()
    texto = ";".join(",".join(str(i) for i in sorted(ids[d == j])) for j in range(5))
    codigo = hashlib.sha256(texto.encode()).hexdigest()[:12]
    assert codigo == "98fac6d202f0", (
        f"o código das dobras deu {codigo}; a parte 1 da aula tem 98fac6d202f0. Veja a ordem: um gerador só, "
        "primeiro as legítimas, depois as fraudes")


def testar_mesmo_sorteio_mesmas_dobras():
    a = np.asarray(dobras_estratificadas(PEQUENO, 3, 5))
    b = np.asarray(dobras_estratificadas(PEQUENO, 3, 5))
    c = np.asarray(dobras_estratificadas(PEQUENO, 3, 6))
    assert (a == b).all(), "o mesmo sorteio deu dobras diferentes"
    assert not (a == c).all(), "os sorteios 5 e 6 deram as mesmas dobras; o sorteio precisa entrar no gerador"


def testar_dobras_demais():
    for n in (1, 8):
        try:
            dobras_estratificadas(PEQUENO, n, 0)
        except ValueError:
            continue
        raise AssertionError(f"aceitou {n} dobras com 7 fraudes; vai de 2 até o número de fraudes")


def testar_rotulo_invalido():
    try:
        dobras_estratificadas(np.array([0, 1, 2, 1, 0, 1]), 2, 0)
    except ValueError:
        return
    raise AssertionError("aceitou o rótulo 2; fraude é 0 ou 1")


def testar_nao_muda_o_rotulo():
    f = PEQUENO.copy()
    dobras_estratificadas(f, 3, 0)
    assert (f == PEQUENO).all(), "dobras_estratificadas mudou o array de fraude que recebeu"


def testar_custo_pequeno():
    r = custo_da_regra(np.array([0, 1, 2, 3, 0]), np.array([1, 1, 0, 1, 0]),
                       np.array([100.0, 200.0, 50.0, 300.0, 80.0]), 2, 10.0)
    esperado = {"marcadas": 2, "pegas": 1, "alarmes": 1, "passou": 300.0, "custo": 320.0}
    for campo, valor in esperado.items():
        assert campo in r, f"falta {campo} no dicionário"
        assert abs(r[campo] - valor) < 1e-9, f"{campo} deu {r[campo]}; o certo é {valor}"


def testar_corte_igual_marca():
    r = custo_da_regra(np.array([1, 0]), np.array([1, 0]), np.array([500.0, 20.0]), 1, 15.0)
    assert r["pegas"] == 1 and r["passou"] == 0, "com 1 voto e corte 1, a compra é marcada (votos >= corte)"


def testar_preco_zero():
    r = custo_da_regra(np.array([0, 3, 0]), np.array([1, 0, 0]), np.array([700.0, 30.0, 40.0]), 2, 0.0)
    assert abs(r["custo"] - 700.0) < 1e-9, f"com a análise de graça, o custo é só a fraude que passou (700); deu {r['custo']}"


def testar_custo_invalido():
    v, f, x = np.array([0, 1]), np.array([0, 1]), np.array([10.0, 20.0])
    casos = [((v, f, x[:1], 1, 15.0), "tamanhos diferentes"), ((v, f, x, 0, 15.0), "corte 0"),
             ((v, f, x, 1, -1.0), "preço negativo")]
    for args, nome in casos:
        try:
            custo_da_regra(*args)
        except ValueError:
            continue
        raise AssertionError(f"custo_da_regra aceitou {nome}; tinha que parar com ValueError")


def testar_custos_da_aula():
    reg = benchmark.benchmark(15.0, 13, dobras_estratificadas, custo_da_regra)
    custos = [round(c) for c in reg["resultados"]["15:1"]["custo"]]
    assert custos == [360, 405, 420, 405, 435], f"K = 15 com pelo menos 1 deu {custos} por dobra; a parte 4 da aula tem 360, 405, 420, 405, 435"
    e = reg["escolha"]
    assert (e["k"], e["corte"]) == (15, 1), f"com R$ 15, a escolha deu K = {e['k']} e corte {e['corte']}; a aula tem K = 15 e 1"


def testar_escrita_no_proprio_arquivo():
    for f in (dobras_estratificadas, custo_da_regra):
        assert f.__module__ == "minha_escolha", f"{f.__name__} veio de {f.__module__}; escreva no minha_escolha.py"
        assert f is not getattr(benchmark, f.__name__), f"{f.__name__} é a do benchmark.py; escreva a sua"
    assert minha_escolha.custo_da_regra is custo_da_regra


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
