"""Exercício da aula «Ridge, Lasso e Elastic Net»: os 40 primeiros pedidos de junho.

Escreva as duas funções abaixo, sem importar as do penalizacao.py. Depois rode, de dentro da
pasta da aula:

    python3 testar_minha_comparacao.py
    python3 minha_comparacao.py

Antes de você escrever as funções, os dois param com NotImplementedError. É o esperado.
"""
import numpy as np
import pandas as pd

from penalizacao import COLUNAS, ajustar, br, carregar, mse, primeiros, tabela


def padronizar(treino, outros):
    """Recebe dois DataFrames com as mesmas colunas de número.

    Retorna (treino_padronizado, outros_padronizado, media, desvio), em que media e desvio são
    Series calculadas só no treino, com o desvio padrão da população (o padrão do np.std).
    Para com ValueError se as colunas dos dois forem diferentes ou se alguma coluna do treino
    não variar. Não muda os DataFrames que recebeu.
    """
    raise NotImplementedError("escreva padronizar(treino, outros)")


def para_reais(coef, intercepto, media, desvio):
    """Leva o resultado de ajustar() nas colunas padronizadas de volta para R$ por unidade.

    coef: os coeficientes padronizados, na ordem de media.index.
    Retorna (Series com R$ por unidade de cada coluna, intercepto em R$).
    """
    raise NotImplementedError("escreva para_reais(coef, intercepto, media, desvio)")


def main():
    dados = carregar()
    treino, outros = primeiros(dados, 6)
    print(f"os 40 primeiros de junho vão de {treino['data_pedido'].min()} a {treino['data_pedido'].max()}; "
          f"os outros {len(outros)} servem para ver o efeito")
    print(f"correlação entre itens e peso_kg nos 40: {br(treino['itens'].corr(treino['peso_kg']), 2)}")
    z, zo, media, desvio = padronizar(treino[COLUNAS], outros[COLUNAS])
    print(f"desvio padrão nos 40: itens {br(desvio['itens'])}, peso_kg {br(desvio['peso_kg'])}")
    print()
    y, yo = treino["valor"].to_numpy(float), outros["valor"].to_numpy(float)

    a = np.column_stack([np.ones(len(dados)), dados[COLUNAS].to_numpy(float)])
    todos = np.linalg.lstsq(a, dados["valor"].to_numpy(float), rcond=None)[0]
    referencia = [todos[j + 1] * desvio[c] for j, c in enumerate(COLUNAS)]

    tabela("A. Ridge: coeficientes padronizados (R$ por desvio padrão da coluna)",
           [("sem", 0, 0), ("0,02", 0, 0.02), ("0,1", 0, 0.1), ("0,5", 0, 0.5), ("2", 0, 2)], z, y, zo, yo)
    tabela("B. Lasso: coeficientes padronizados",
           [("sem", 0, 0), ("5", 5, 0), ("10", 10, 0), ("20", 20, 0), ("40", 40, 0)], z, y, zo, yo)
    ajustes = tabela("C. Os três lado a lado, e todos os pedidos sem penalidade (na mesma escala)",
                     [("sem", 0, 0), ("Ridge", 0, 0.1), ("Lasso", 20, 0), ("Elastic", 15, 0.2)],
                     z, y, zo, yo, referencia=referencia)

    print("D. Os mesmos ajustes da parte C em R$ por unidade")
    reais = [para_reais(b, i, media, desvio) for b, i in ajustes]
    print(f"  {'coluna':22}" + "".join(f"{n:>11}" for n in ("sem", "Ridge", "Lasso", "Elastic", "todos")))
    print(f"  {'intercepto':22}" + "".join(f"{br(i):>11}" for _, i in reais) + f"{br(todos[0]):>11}")
    for j, c in enumerate(COLUNAS):
        print(f"  {c:22}" + "".join(f"{br(r[c]):>11}" for r, _ in reais) + f"{br(todos[j + 1]):>11}")
    for (b, i), (r, ir) in zip(ajustes, reais):
        x = outros[COLUNAS].to_numpy(float)
        if abs(mse(x, yo, r.to_numpy(float), ir) - mse(zo, yo, b, i)) > 1e-6:
            raise AssertionError("o modelo em R$ por unidade não prevê igual ao padronizado")
    print("  em R$ por unidade, os quatro modelos preveem os outros pedidos igual à parte C")


if __name__ == "__main__":
    main()
