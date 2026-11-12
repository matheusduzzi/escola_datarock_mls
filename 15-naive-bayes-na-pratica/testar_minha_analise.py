"""Testes das funções matriz_confusao e cobertura do minha_analise.py.

Rode de dentro da pasta da aula:  python3 testar_minha_analise.py
Os testes usam casos pequenos feitos à mão e a regra de palavras em junho, que a aula já mostra.
A leitura de julho continua com você.
"""
import minha_analise
from filas import ASSUNTOS, carregar, palavras, regra, separar, treinar
from minha_analise import cobertura, matriz_confusao


def testar_caso_pequeno():
    m = matriz_confusao(["atraso", "atraso", "troca"], ["atraso", "troca", "troca"])
    assert m["atraso"]["atraso"] == 1 and m["atraso"]["troca"] == 1 and m["troca"]["troca"] == 1, \
        f"veio {m}; uma mensagem de atraso ficou em atraso, uma de atraso foi para troca e uma de troca ficou em troca"


def testar_linha_e_o_marcado():
    m = matriz_confusao(["pagamento"], ["outro"])
    assert m["pagamento"]["outro"] == 1, "a linha é o assunto marcado e a coluna o previsto: m['pagamento']['outro'] = 1"
    assert m["outro"]["pagamento"] == 0, "m['outro']['pagamento'] precisa ser 0; você trocou linha e coluna?"


def testar_todos_os_assuntos_com_zero():
    m = matriz_confusao(["troca"], ["troca"])
    assert list(m) == ASSUNTOS, f"linhas {list(m)}; precisam ser {ASSUNTOS}, nessa ordem"
    for a in ASSUNTOS:
        assert list(m[a]) == ASSUNTOS, f"colunas da linha {a}: {list(m[a])}; precisam ser {ASSUNTOS}"
    assert sum(m[a][b] for a in ASSUNTOS for b in ASSUNTOS) == 1, "só uma mensagem; o resto da matriz é 0"


def testar_outra_lista_de_assuntos():
    m = matriz_confusao(["b", "a"], ["b", "b"], assuntos=["a", "b"])
    assert list(m) == ["a", "b"] and m["a"]["b"] == 1 and m["b"]["b"] == 1, f"com assuntos=['a', 'b'] veio {m}"


def testar_tamanhos_diferentes():
    try:
        matriz_confusao(["atraso", "troca"], ["atraso"])
    except ValueError:
        return
    raise AssertionError("2 marcados e 1 previsto precisam parar com ValueError")


def testar_assunto_fora_da_lista():
    for reais, previstos in [(["atraso"], ["entrega"]), (["Atraso"], ["atraso"])]:
        try:
            matriz_confusao(reais, previstos)
        except ValueError:
            continue
        raise AssertionError(f"{reais} e {previstos} têm assunto fora da lista; precisa parar com ValueError")


def testar_nao_muda_o_que_recebeu():
    reais, previstos = ["troca", "atraso"], ["atraso", "atraso"]
    matriz_confusao(reais, previstos)
    assert reais == ["troca", "atraso"] and previstos == ["atraso", "atraso"], "a função mudou as listas recebidas"


def testar_regra_em_junho():
    _, junho = separar(carregar())
    reais = [l["assunto"] for l in junho]
    m = matriz_confusao(reais, [regra(palavras(l["texto"]))[0] for l in junho])
    acertos = sum(m[a][a] for a in ASSUNTOS)
    assert acertos == 279, f"a regra acerta 279 das 321 de junho (parte 4 da aula); a diagonal deu {acertos}"
    assert m["atraso"]["outro"] == 11, f"atraso -> outro são 11 em junho (parte 4 da aula); deu {m['atraso']['outro']}"
    for a in ASSUNTOS:
        assert sum(m[a].values()) == reais.count(a), f"a linha {a} precisa somar as {reais.count(a)} mensagens de {a}"


def testar_cobertura_simples():
    vocab = ["chegou", "nao", "pedido"]
    assert cobertura(["pedido", "nao", "chegou"], vocab) == 1.0, "as três palavras estão no vocabulário: 1.0"
    assert cobertura(["motoboy", "nao"], vocab) == 0.5, "uma de duas palavras no vocabulário: 0.5"


def testar_cobertura_conta_repeticao():
    c = cobertura(["xx", "xx", "nao"], {"nao"})
    assert abs(c - 1 / 3) < 1e-9, f"deu {c}; a palavra repetida conta cada vez: 1 de 3 palavras"


def testar_cobertura_vazia():
    assert cobertura([], ["nao"]) == 0.0, "mensagem sem palavra dá 0.0"


def testar_cobertura_recusa_texto():
    try:
        cobertura("nao chegou", ["nao", "chegou"])
    except TypeError:
        return
    raise AssertionError("cobertura recebeu um texto; precisa parar com TypeError")


def testar_cobertura_com_o_modelo():
    treino, _ = separar(carregar())
    vocab = treinar(treino)["vocab"]
    c = cobertura(palavras("o carteiro não apareceu"), vocab)
    assert c == 0.75, f"«o carteiro não apareceu»: só «carteiro» falta no vocabulário de janeiro a maio, 3 de 4; deu {c}"


def testar_escrita_no_proprio_arquivo():
    for f in (matriz_confusao, cobertura):
        assert f.__module__ == "minha_analise", f"{f.__name__} veio de {f.__module__}; escreva no minha_analise.py"
    assert minha_analise.matriz_confusao is matriz_confusao and minha_analise.cobertura is cobertura


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
