"""Testes das funções matriz, por_grupo, salvar e carregar do meu_modelo.py.

Rode de dentro da pasta da aula:  python3 testar_meu_modelo.py
Os testes usam seis pedidos inventados aqui, não os pedidos de abril e maio: eles mostram que o
código faz a conta certa, e a leitura para a gerente continua com você.
O teste de salvar e carregar grava numa pasta temporária, que some no fim.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from meu_modelo import carregar, matriz, por_grupo, salvar

CANCELOU = [1, 0, 1, 0, 0, 1]
LIGAR = [1, 1, 0, 0, 1, 1]
FORMA = ["pix", "boleto", "boleto", "pix", "cartao", "boleto"]
MODELO = {"colunas": ["valor", "canal_app"], "pesos": [-1.5, 0.25, 0.75]}


def para_ver_erro(funcao, *args):
    try:
        funcao(*args)
    except ValueError:
        return True
    return False


def testar_as_quatro_casas():
    m = matriz(CANCELOU, LIGAR)
    assert m == {"vp": 2, "fp": 2, "fn": 1, "vn": 1}, f"veio {m}; eram VP 2, FP 2, FN 1, VN 1"


def testar_numeros_inteiros():
    m = matriz(np.array(CANCELOU), np.array(LIGAR, dtype=bool))
    assert all(type(v) is int for v in m.values()), f"as casas precisam ser int do Python, vieram {m}"
    assert m == {"vp": 2, "fp": 2, "fn": 1, "vn": 1}, f"com True e False no ligar veio {m}"


def testar_tamanhos_diferentes():
    assert para_ver_erro(matriz, CANCELOU, LIGAR[:4]), "matriz aceitou 6 resultados e 4 decisões"
    assert para_ver_erro(matriz, CANCELOU, [1]), "matriz aceitou 6 resultados e 1 decisão; o NumPy repete a decisão sem avisar"


def testar_nota_no_lugar_de_ligar():
    nota = [0.9, 0.4, 0.2, 0.1, 0.6, 0.8]
    assert para_ver_erro(matriz, CANCELOU, nota), "matriz aceitou notas no lugar da decisão de ligar; compare com o corte antes"


def testar_y_fora_de_zero_e_um():
    assert para_ver_erro(matriz, [1, 0, 2, 0, 0, 1], LIGAR), "matriz aceitou o valor 2 em y"


def testar_grupos_em_ordem():
    g = por_grupo(CANCELOU, LIGAR, FORMA)
    assert list(g) == ["boleto", "cartao", "pix"], f"grupos {list(g)}; eram boleto, cartao, pix"
    assert g["boleto"] == {"vp": 1, "fp": 1, "fn": 1, "vn": 0}, f"boleto veio {g['boleto']}"
    assert g["cartao"] == {"vp": 0, "fp": 1, "fn": 0, "vn": 0}, f"cartao veio {g['cartao']}"


def testar_grupos_somam_o_total():
    g = por_grupo(CANCELOU, LIGAR, FORMA)
    soma = {k: sum(m[k] for m in g.values()) for k in ("vp", "fp", "fn", "vn")}
    assert soma == matriz(CANCELOU, LIGAR), f"a soma dos grupos deu {soma}, diferente da matriz total"


def testar_grupo_de_outro_tamanho():
    assert para_ver_erro(por_grupo, CANCELOU, LIGAR, FORMA[:5]), "por_grupo aceitou 6 pedidos e 5 grupos"


def testar_salvar_e_carregar():
    with tempfile.TemporaryDirectory() as pasta:
        arquivo = Path(pasta) / "modelo.json"
        salvar(MODELO, arquivo)
        de_volta = carregar(arquivo)
        assert de_volta["colunas"] == MODELO["colunas"], f"colunas voltaram {de_volta['colunas']}"
        assert np.allclose(de_volta["pesos"], MODELO["pesos"]), f"pesos voltaram {de_volta['pesos']}"
        gravado = json.loads(arquivo.read_text(encoding="utf-8"))
        assert set(gravado) == {"colunas", "pesos"}, f"o JSON tem {sorted(gravado)}; eram colunas e pesos"


def testar_salvar_pesos_errados_nao_grava():
    with tempfile.TemporaryDirectory() as pasta:
        for errado in ({"colunas": ["valor", "canal_app"], "pesos": [0.25, 0.75]},
                       {"colunas": ["valor", "canal_app"], "pesos": [-1.5, float("nan"), 0.75]}):
            arquivo = Path(pasta) / "modelo.json"
            assert para_ver_erro(salvar, errado, arquivo), f"salvar aceitou {errado}"
            assert not arquivo.exists(), f"salvar parou com {errado}, mas gravou o arquivo mesmo assim"


def testar_carregar_arquivo_mexido():
    with tempfile.TemporaryDirectory() as pasta:
        arquivo = Path(pasta) / "modelo.json"
        for mexido in ({"colunas": ["valor"], "pesos": [-1.5, 0.25, 0.75]}, {"pesos": [-1.5, 0.25, 0.75]}):
            arquivo.write_text(json.dumps(mexido), encoding="utf-8")
            assert para_ver_erro(carregar, arquivo), f"carregar aceitou o arquivo {mexido}"


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
        except Exception as erro:
            falhas += 1
            print(f"FALHOU {nome}: {type(erro).__name__}: {erro}")
    if falhas:
        print(f"{falhas} de {len(testes)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(testes)} testes passaram")
