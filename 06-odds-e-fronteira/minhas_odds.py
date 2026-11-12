"""Exercício da aula «Odds e fronteira»: a regra com cliente novo que a Lia recebeu.

Escreva as duas funções marcadas com NotImplementedError. Depois rode, de dentro da pasta da aula:
    python3 testar_minhas_odds.py
    python3 minhas_odds.py
"""
import math
import sys

import numpy as np
import pandas as pd

BASE = '../dados/loja/pedidos.csv'

# A regra da Lia: score = A + B_BOLETO × boleto + B_PRAZO × prazo + B_NOVO × novo
A, B_BOLETO, B_PRAZO, B_NOVO = -3.12, 1.18, 0.108, 1.09
CORTE = 0.30

# Os dois pedidos da tarefa 1: (nome, boleto, prazo), os dois de cliente antigo
PEDIDOS = [('A', 0, 3), ('B', 1, 14)]
GRUPOS = [('antigo, sem boleto', 0, 0), ('antigo, com boleto', 0, 1),
          ('novo, sem boleto', 1, 0), ('novo, com boleto', 1, 1)]


def br(x, casas=4):
    return f'{x:.{casas}f}'.replace('.', ',')


def nota(score):
    """A sigmoide da aula 05, pronta."""
    return 1 / (1 + math.exp(-score))


def comparar(score_base, b):
    """O que acontece quando o score sai de score_base e sobe b.

    Retorna a tupla (nota_antes, nota_depois, pontos, razao_chances, razao_odds):
    as duas notas, a diferença em pontos (100 × (depois − antes)), a razão das notas
    (depois ÷ antes) e a razão das odds (odds depois ÷ odds antes).
    """
    raise NotImplementedError('escreva comparar(score_base, b)')


def prazo_da_fronteira(corte, a, b_prazo, resto=0.0):
    """O prazo em que a nota fica igual ao corte, para score = a + resto + b_prazo × prazo.

    resto é a soma das outras parcelas do score (por exemplo, B_BOLETO × boleto + B_NOVO × novo).
    Para com ValueError se corte não estiver entre 0 e 1 (sem incluir 0 e 1) ou se b_prazo for 0.
    """
    raise NotImplementedError('escreva prazo_da_fronteira(corte, a, b_prazo, resto)')


def main():
    print(f'Regra da Lia: score = {br(A, 2)} + {br(B_BOLETO, 2)} × boleto + {br(B_PRAZO, 3)} × prazo + {br(B_NOVO, 2)} × novo')
    print(f'razão de odds do cliente novo: {br(math.exp(B_NOVO), 3)}')
    print()
    print('1. Os pedidos A e B, como cliente antigo e como cliente novo')
    print('   pedido  boleto  prazo   antigo    novo   pontos  razão das chances  razão das odds')
    for nome, bol, prazo in PEDIDOS:
        antes, depois, pontos, rc, ro = comparar(A + B_BOLETO * bol + B_PRAZO * prazo, B_NOVO)
        print(f'   {nome:<6}  {bol:>6}  {prazo:>5}  {br(antes):>7}  {br(depois):>6}  {br(pontos, 1):>7}  {br(rc, 2):>17}  {br(ro, 3):>14}')
    print()

    try:
        pedidos = pd.read_csv(BASE)
    except FileNotFoundError:
        sys.exit(f"No such file or directory: '{BASE}'. Rode de dentro da pasta da aula, com a pasta dados/ do kit ao lado.")
    treino = pedidos[pedidos['data_pedido'] < '2026-04-01']
    novo = treino['cliente_novo'].to_numpy()
    boleto = (treino['forma_pagamento'] == 'boleto').to_numpy().astype(int)
    prazo = treino['prazo_prometido_dias'].to_numpy()
    cancelou = treino['cancelado'].to_numpy()

    print(f'2. A fronteira com corte {br(CORTE, 2)}, nos {len(treino)} pedidos de janeiro a março')
    print('   grupo                prazo da fronteira  pedidos  acima do corte  cancelaram acima')
    for rotulo, nv, bol in GRUPOS:
        f = prazo_da_fronteira(CORTE, A, B_PRAZO, B_BOLETO * bol + B_NOVO * nv)
        m = (novo == nv) & (boleto == bol)
        acima = m & (prazo > f)
        print(f'   {rotulo:<19}  {br(f, 2):>18}  {int(m.sum()):>7}  {int(acima.sum()):>14}  {int(cancelou[acima].sum()):>16}')


if __name__ == '__main__':
    main()
