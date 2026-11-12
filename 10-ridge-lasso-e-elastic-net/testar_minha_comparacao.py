"""Testes das funções padronizar e para_reais do minha_comparacao.py.

Rode de dentro da pasta da aula:  python3 testar_minha_comparacao.py
Os testes usam os cinco pedidos e os 40 pedidos de janeiro da aula, não os de junho: eles
mostram que o código faz a conta da aula, e a leitura de junho continua com você.
"""
import numpy as np
import pandas as pd

import minha_comparacao
from minha_comparacao import padronizar, para_reais
from penalizacao import COLUNAS, ajustar, carregar, primeiros

PESO = pd.DataFrame({"peso_kg": [1.0, 2.0, 3.0, 4.0, 5.0], "itens": [1.0, 1.0, 2.0, 2.0, 4.0]})


def perto(a, b, tol=1e-9):
    return abs(a - b) < tol


def testar_treino_com_media_zero_e_desvio_um():
    zt, _, media, desvio = padronizar(PESO, PESO.head(2))
    for c in PESO.columns:
        assert perto(zt[c].mean(), 0), f"média de {c} padronizada deu {zt[c].mean()}, tinha que dar 0"
        assert perto(zt[c].std(ddof=0), 1), f"desvio de {c} padronizado deu {zt[c].std(ddof=0)}, tinha que dar 1"
    assert perto(media["peso_kg"], 3) and perto(desvio["peso_kg"], 2 ** 0.5), "média ou desvio do peso errados"


def testar_desvio_da_populacao():
    zt, _, _, _ = padronizar(PESO, PESO)
    assert perto(zt["peso_kg"].iloc[-1], 2 / 2 ** 0.5), (
        f"o pedido de 5 kg deu {zt['peso_kg'].iloc[-1]:.4f}; com o desvio da população (np.std) dá 1,4142")


def testar_outros_usam_a_media_do_treino():
    outros = pd.DataFrame({"peso_kg": [3.0, 5.0, 7.0], "itens": [2.0, 2.0, 2.0]})
    _, zo, _, _ = padronizar(PESO, outros)
    assert perto(zo["peso_kg"].iloc[0], 0), "3 kg é a média do treino e tinha que virar 0"
    assert perto(zo["peso_kg"].iloc[2], 4 / 2 ** 0.5), (
        "7 kg tinha que virar (7 - 3) ÷ 1,4142 = 2,8284, com a média e o desvio do treino")


def testar_coluna_sem_variacao():
    parado = PESO.assign(cupom=0.0)
    try:
        padronizar(parado, parado)
    except ValueError:
        return
    raise AssertionError("padronizar aceitou uma coluna que não varia no treino; tinha que parar com ValueError")


def testar_colunas_diferentes():
    try:
        padronizar(PESO, PESO[["itens", "peso_kg"]])
    except ValueError:
        return
    raise AssertionError("padronizar aceitou outros com as colunas em outra ordem; tinha que parar com ValueError")


def testar_nao_muda_a_entrada():
    copia = PESO.copy()
    padronizar(PESO, PESO)
    assert PESO.equals(copia), "padronizar mudou o DataFrame que recebeu"


def testar_volta_para_reais_igual_a_regressao_direta():
    # Sem penalidade, padronizar e voltar tem que dar a regressão nas colunas como estão: aula, parte 1.
    treino, _ = primeiros(carregar(), 1)
    z, _, media, desvio = padronizar(treino[COLUNAS], treino[COLUNAS])
    coef, intercepto = ajustar(z, treino["valor"])
    por_unidade, inter = para_reais(coef, intercepto, media, desvio)
    a = np.column_stack([np.ones(len(treino)), treino[COLUNAS].to_numpy(float)])
    direto = np.linalg.lstsq(a, treino["valor"].to_numpy(float), rcond=None)[0]
    assert perto(inter, direto[0], 1e-6), f"intercepto em R$ deu {inter:.2f}, a aula tem {direto[0]:.2f}"
    for j, c in enumerate(COLUNAS):
        assert perto(por_unidade[c], direto[j + 1], 1e-6), (
            f"{c} em R$ por unidade deu {por_unidade[c]:.2f}, a aula tem {direto[j + 1]:.2f}")


def testar_volta_com_penalidade_preve_igual():
    treino, outros = primeiros(carregar(), 1)
    z, zo, media, desvio = padronizar(treino[COLUNAS], outros[COLUNAS])
    coef, intercepto = ajustar(z, treino["valor"], 20, 0)
    por_unidade, inter = para_reais(coef, intercepto, media, desvio)
    previsto_z = intercepto + zo.to_numpy(float).dot(coef)
    previsto_r = inter + outros[COLUNAS].to_numpy(float).dot(por_unidade[COLUNAS].to_numpy(float))
    assert np.allclose(previsto_z, previsto_r), "em R$ por unidade, o Lasso alfa 20 prevê diferente do padronizado"


def testar_escrita_no_proprio_arquivo():
    for f in (padronizar, para_reais):
        assert f.__module__ == "minha_comparacao", f"{f.__name__} veio de {f.__module__}; escreva no minha_comparacao.py"
    assert minha_comparacao.padronizar is padronizar


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
