"""Exercício da aula «KNN e dimensionalidade»: as compras de junho, com a hora de dois jeitos.

Escreva as duas funções abaixo, sem importar as do vizinhos.py. Depois rode, de dentro da
pasta da aula:

    python3 testar_meus_vizinhos.py
    python3 meus_vizinhos.py

Antes de você escrever as funções, os dois param com NotImplementedError. É o esperado.
"""
import math

import numpy as np
import pandas as pd

from vizinhos import COLUNAS, br, carregar, separar


def distancias(conhecidas, nova, desvio):
    """Distância de uma compra nova a cada compra conhecida, com as colunas padronizadas.

    conhecidas: DataFrame, uma linha por compra conhecida.
    nova: Series com as mesmas colunas, na mesma ordem.
    desvio: Series com o desvio padrão de cada coluna, na mesma ordem.
    Retorna um array do NumPy com uma distância por linha de conhecidas, na ordem das linhas:
    a raiz da soma dos quadrados de (coluna da conhecida − coluna da nova) ÷ desvio da coluna.
    Para com ValueError se as colunas não forem as mesmas nos três ou se algum desvio for zero
    ou negativo. Não muda o que recebeu.
    """
    raise NotImplementedError("escreva distancias(conhecidas, nova, desvio)")

def prever(dist, rotulos, k):
    """Vota com os k vizinhos mais perto.

    dist: as distâncias de distancias(); rotulos: 0 ou 1 de cada compra conhecida, na mesma ordem.
    Retorna (previsao, votos): votos é quantos dos k vizinhos foram fraude; previsao é 1 quando
    eles são mais da metade, 0 nos outros casos. Em empate de distância, fica a compra que vem
    primeiro. Para com ValueError se k não for um número ímpar entre 1 e o número de compras.
    """
    raise NotImplementedError("escreva prever(dist, rotulos, k)")

# ---- daqui para baixo já está pronto ----

def com_relogio(tabela):
    """Troca a coluna hora por duas: a posição da hora num relógio de ponteiro (x e y)."""
    angulo = 2 * math.pi * tabela["hora"] / 24
    nova = tabela.drop(columns="hora")
    return nova.assign(hora_x=np.cos(angulo).round(6), hora_y=np.sin(angulo).round(6))


def rodar(conh, novas, colunas, k=5):
    desvio = conh[colunas].std(ddof=0)
    rc = conh["fraude"].to_numpy()
    marcadas, falsas, escaparam = 0, 0, []
    for _, compra in novas.iterrows():
        previsao, _ = prever(distancias(conh[colunas], compra[colunas], desvio), rc, k)
        if previsao and compra["fraude"]:
            marcadas += 1
        elif previsao:
            falsas += 1
        elif compra["fraude"]:
            escaparam.append(compra)
    return marcadas, falsas, escaparam


def mostrar(rotulo, conh, novas, colunas):
    marcadas, falsas, escaparam = rodar(conh, novas, colunas)
    total = int(novas["fraude"].sum())
    legit = "compra legítima" if falsas == 1 else "compras legítimas"
    print(f"{rotulo}")
    print(f"  colunas: {', '.join(colunas)}")
    print(f"  K = 5: marca {marcadas} das {total} fraudes de junho e {falsas} {legit}")
    print(f"  fraudes que escaparam: {len(escaparam)}")
    for c in escaparam:
        print(f"    compra {c['id_transacao']}: valor {br(c['valor'])}, conta {c['idade_conta_dias']} dias, "
              f"{c['tentativas_cartao']} {'tentativa' if c['tentativas_cartao'] == 1 else 'tentativas'}, entrega a {br(c['distancia_entrega_km'], 1)} km, "
              f"{c['hora']}h")


def main():
    dados = carregar()
    conh, novas = separar(dados)
    print(f"{len(conh)} compras conhecidas (janeiro a maio) e {len(novas)} compras de junho")
    print()
    mostrar("A. A hora como número, de 0 a 23", conh, novas, COLUNAS)
    print()
    rc, rn = com_relogio(conh), com_relogio(novas)
    colunas_b = [c for c in COLUNAS if c != "hora"] + ["hora_x", "hora_y"]
    mostrar("B. A hora como posição no relógio", rc, rn, colunas_b)
    print()
    print("C. Só a parte da hora na distância, em desvios padrão")
    d_num = conh["hora"].std(ddof=0)
    d_rel = rc[["hora_x", "hora_y"]].std(ddof=0)
    print(f"  desvio de janeiro a maio: hora {br(d_num)}, hora_x {br(d_rel['hora_x'])}, hora_y {br(d_rel['hora_y'])}")
    print("  de       até     como número   no relógio")
    for de, ate in ((23, 0), (23, 3), (23, 12), (3, 12)):
        a = com_relogio(pd.DataFrame({"hora": [de, ate]}))
        rel = math.sqrt(sum(((a[c].iloc[0] - a[c].iloc[1]) / d_rel[c]) ** 2 for c in ("hora_x", "hora_y")))
        print(f"  {de:>2}h      {ate:>2}h     {br(abs(de - ate) / d_num):>11} {br(rel):>12}")


if __name__ == "__main__":
    main()
