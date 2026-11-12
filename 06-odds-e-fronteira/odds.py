"""As contas da aula «Odds e fronteira»: odds a partir da contagem, o b lido em odds,
a mesma razão de odds com mudanças diferentes na chance e a fronteira de cada corte.

Rode de dentro da pasta da aula:  python3 odds.py
Lê ../dados/loja/pedidos.csv e usa só os pedidos de janeiro a março.
"""
import math
import sys

import numpy as np
import pandas as pd

BASE = '../dados/loja/pedidos.csv'
TAXA = 0.05
PASSOS = 20000


def br(x, casas=4):
    """Número com vírgula decimal, como no texto da aula."""
    return f'{x:.{casas}f}'.replace('.', ',')


def nota(score):
    return 1 / (1 + np.exp(-np.asarray(score, dtype=float)))


def logit(p):
    return math.log(p / (1 - p))


def ajustar(colunas, y):
    """Os passos do gradiente da aula 05, saindo de zero, para score = a + b1 × x1 + b2 × x2 + ..."""
    a = 0.0
    b = [0.0] * len(colunas)
    for _ in range(PASSOS):
        score = a + sum(bi * x for bi, x in zip(b, colunas))
        r = nota(score) - y
        a -= TAXA * r.mean()
        b = [bi - TAXA * (r * x).mean() for bi, x in zip(b, colunas)]
    score = a + sum(bi * x for bi, x in zip(b, colunas))
    r = nota(score) - y
    maior_derivada = max([abs(r.mean())] + [abs((r * x).mean()) for x in colunas])
    return a, b, maior_derivada


def ler():
    try:
        pedidos = pd.read_csv(BASE)
    except FileNotFoundError:
        sys.exit(f"No such file or directory: '{BASE}'. Rode de dentro da pasta da aula, com a pasta dados/ do kit ao lado.")
    treino = pedidos[pedidos['data_pedido'] < '2026-04-01']
    y = treino['cancelado'].to_numpy(float)
    boleto = (treino['forma_pagamento'] == 'boleto').to_numpy(float)
    prazo = treino['prazo_prometido_dias'].to_numpy(float)
    novo = treino['cliente_novo'].to_numpy(float)
    return y, boleto, prazo, novo


def main():
    y, boleto, prazo, novo = ler()
    n, canc = len(y), int(y.sum())

    print('1. Chance e odds nos pedidos de janeiro a março')
    print(f'   {n} pedidos, {canc} cancelaram, {n - canc} foram entregues')
    print(f'   chance: {canc} ÷ {n} = {br(canc / n)} | odds: {canc} ÷ {n - canc} = {br(canc / (n - canc))}')
    print()

    print('2. Cliente novo e cliente antigo, na contagem')
    print('   cliente    pedidos  cancelaram  entregues  chance    odds')
    linhas = {}
    for rotulo, valor in (('novo', 1), ('antigo', 0)):
        m = novo == valor
        c, t = int(y[m].sum()), int(m.sum())
        linhas[rotulo] = (c / t, c / (t - c))
        print(f'   {rotulo:<7}  {t:>9}  {c:>10}  {t - c:>9}  {br(c / t)}  {br(c / (t - c))}')
    (pn, on), (pa, oa) = linhas['novo'], linhas['antigo']
    print(f'   razão das odds: {br(on)} ÷ {br(oa)} = {br(on / oa, 3)}')
    print(f'   razão das chances: {br(pn)} ÷ {br(pa)} = {br(pn / pa, 3)}')
    print(f'   diferença das chances: {br(100 * (pn - pa), 1)} pontos')
    print()

    print(f'3. Os modelos, com {PASSOS} passos do gradiente da aula 05 (taxa {br(TAXA, 2)}), saindo de zero')
    a1, (b1,), d1 = ajustar([novo], y)
    print(f'   só cliente novo:    score = {br(a1)} + {br(b1)} × novo'
          f'   | e elevado a {br(b1)} = {br(math.exp(b1), 3)}')
    a, (bb, bp), d2 = ajustar([boleto, prazo], y)
    print(f'   boleto e prazo:     score = {br(a)} + {br(bb)} × boleto + {br(bp)} × prazo')
    a3, (bb3, bp3, bn3), d3 = ajustar([boleto, prazo, novo], y)
    print(f'   com cliente novo:   score = {br(a3)} + {br(bb3)} × boleto + {br(bp3)} × prazo + {br(bn3)} × novo')
    print(f'   maior derivada no fim dos três: {max(d1, d2, d3):.1e}'.replace('.', ','))
    print()

    print('4. O b lido em odds (modelo com boleto e prazo)')
    print(f'   boleto: e elevado a {br(bb)} = {br(math.exp(bb), 3)}, com o mesmo prazo')
    print(f'   prazo: e elevado a {br(bp)} = {br(math.exp(bp), 4)} por dia;'
          f' 5 dias: e elevado a {br(5 * bp)} = {br(math.exp(5 * bp), 3)}')
    print()

    print('5. A mesma razão de odds, mudanças diferentes na chance')
    print('   prazo   sem boleto   com boleto   diferença (pontos)   razão das chances   razão das odds')
    for d in (2, 5, 8, 12, 15):
        p0, p1 = float(nota(a + bp * d)), float(nota(a + bb + bp * d))
        razao_odds = (p1 / (1 - p1)) / (p0 / (1 - p0))
        print(f'   {d:>5}   {br(p0):>10}   {br(p1):>10}   {br(100 * (p1 - p0), 1):>18}'
              f'   {br(p1 / p0, 2):>17}   {br(razao_odds, 3):>14}')
    print()

    print(f'6. A razão de odds {br(math.exp(bb), 3)} aplicada a chances de partida diferentes')
    print('   chance antes   odds antes   odds depois   chance depois   razão das chances')
    for p0 in (0.01, 0.05, 0.2, 0.5, 0.8):
        o1 = p0 / (1 - p0) * math.exp(bb)
        p1 = o1 / (1 + o1)
        print(f'   {br(p0, 2):>12}   {br(p0 / (1 - p0)):>10}   {br(o1):>11}   {br(p1):>13}   {br(p1 / p0, 2):>17}')
    print()

    print('7. Boleto na contagem, por faixa de prazo')
    print('   prazo   com boleto (canc/ped)   sem boleto (canc/ped)   razão das odds na contagem')
    for lo, hi in ((2, 5), (6, 10), (11, 15)):
        m = (prazo >= lo) & (prazo <= hi)
        cb, tb = int(y[m & (boleto == 1)].sum()), int((m & (boleto == 1)).sum())
        cs, ts = int(y[m & (boleto == 0)].sum()), int((m & (boleto == 0)).sum())
        ro = (cb / (tb - cb)) / (cs / (ts - cs))
        print(f'   {lo:>2} a {hi:<2}   {cb:>12}/{tb:<8}   {cs:>12}/{ts:<8}   {br(ro, 2):>26}')
    print(f'   o modelo, em qualquer prazo: {br(math.exp(bb), 3)}')
    print()

    print('8. A fronteira: onde a nota passa do corte (modelo com boleto e prazo)')
    print(f'   boleto vale {br(bb)} ÷ {br(bp)} = {br(bb / bp, 1)} dias de prazo')
    print('   corte   score na fronteira   prazo da fronteira (sem boleto | com boleto)   pedidos acima   cancelaram')
    score = a + bb * boleto + bp * prazo
    for corte in (0.5, 0.25):
        z = logit(corte)
        f0, f1 = (z - a) / bp, (z - a - bb) / bp
        acima = nota(score) >= corte
        print(f'   {br(corte, 2):>5}   {br(z):>18}   {br(f0, 1):>21} | {br(f1, 1):<21}   {int(acima.sum()):>13}   {int(y[acima].sum()):>10}')


if __name__ == '__main__':
    main()
