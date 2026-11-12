"""Testes das funções do minhas_odds.py, com os números da aula (não os da Lia).

Rode de dentro da pasta da aula:  python3 testar_minhas_odds.py
"""
import math
import sys

from minhas_odds import comparar, prazo_da_fronteira

# O modelo da aula: score = -2,6542 + 1,0913 × boleto + 0,1112 × prazo
A, B_BOLETO, B_PRAZO = -2.6542, 1.0913, 0.1112

falhas = []
total = 0


def perto(nome, saiu, esperado, tol=1e-3):
    global total
    total += 1
    if not (isinstance(saiu, (int, float)) and math.isfinite(saiu) and abs(saiu - esperado) <= tol):
        falhas.append(f'{nome}: saiu {saiu!r}, a aula tem {esperado}')


def para_com_valueerror(nome, funcao, *args):
    global total
    total += 1
    try:
        funcao(*args)
    except ValueError:
        return
    except Exception as erro:  # outro erro também é falha: o enunciado pede ValueError
        falhas.append(f'{nome}: parou com {type(erro).__name__}, o enunciado pede ValueError')
        return
    falhas.append(f'{nome}: não parou; o enunciado pede ValueError')


# 1. comparar: prazo 8 sem boleto e com boleto (seção 5 da aula)
r = comparar(A + B_PRAZO * 8, B_BOLETO)
if not (isinstance(r, tuple) and len(r) == 5):
    print(f'falhou: comparar precisa retornar uma tupla com 5 números; saiu {r!r}')
    sys.exit(1)
antes, depois, pontos, rc, ro = r
perto('nota antes, prazo 8 sem boleto', antes, 0.1462, 1e-4)
perto('nota depois, prazo 8 com boleto', depois, 0.3377, 1e-4)
perto('pontos, prazo 8', pontos, 19.15, 0.01)
perto('razão das chances, prazo 8', rc, 2.31, 0.005)
perto('razão das odds, prazo 8', ro, math.exp(B_BOLETO), 1e-6)

# 2. A razão das odds é a mesma em qualquer base; os pontos, não
_, _, pontos2, _, ro2 = comparar(A + B_PRAZO * 2, B_BOLETO)
_, _, pontos15, _, ro15 = comparar(A + B_PRAZO * 15, B_BOLETO)
perto('pontos, prazo 2', pontos2, 12.66, 0.01)
perto('pontos, prazo 15', pontos15, 25.46, 0.01)
perto('razão das odds, prazo 15', ro15, math.exp(B_BOLETO), 1e-6)

# 3. b negativo: a chance cai e a razão das odds fica abaixo de 1
_, depois_neg, pontos_neg, _, ro_neg = comparar(0.0, -1.0)
perto('nota depois, score 0 e b = -1', depois_neg, 0.2689, 1e-4)
perto('pontos, score 0 e b = -1', pontos_neg, -23.11, 0.01)
perto('razão das odds, b = -1', ro_neg, math.exp(-1), 1e-6)

# 4. prazo_da_fronteira (seção 8 da aula)
perto('fronteira, corte 0,5, com boleto', prazo_da_fronteira(0.5, A, B_PRAZO, B_BOLETO), 14.055, 0.001)
perto('fronteira, corte 0,5, sem boleto', prazo_da_fronteira(0.5, A, B_PRAZO), 23.869, 0.001)
perto('fronteira, corte 0,25, sem boleto', prazo_da_fronteira(0.25, A, B_PRAZO, 0.0), 13.989, 0.001)

# 5. O que o enunciado manda recusar
para_com_valueerror('corte 0', prazo_da_fronteira, 0.0, A, B_PRAZO)
para_com_valueerror('corte 1', prazo_da_fronteira, 1.0, A, B_PRAZO)
para_com_valueerror('corte 30 (em vez de 0,30)', prazo_da_fronteira, 30, A, B_PRAZO)
para_com_valueerror('corte negativo', prazo_da_fronteira, -0.2, A, B_PRAZO)
para_com_valueerror('b_prazo zero', prazo_da_fronteira, 0.5, A, 0.0)

if falhas:
    print(f'falhou: {len(falhas)} de {total} testes')
    for f in falhas:
        print('  ' + f)
    sys.exit(1)
print(f'ok: {total} testes passaram')
