"""Exercício da aula «Naive Bayes»: as tentativas do cartão como categoria.

Escreva a função categorica_log abaixo, sem importar nada que já faça a conta. Depois rode, de
dentro da pasta da aula:

    python3 testar_meu_bayes.py
    python3 meu_bayes.py

Antes de você escrever a função, os dois param com NotImplementedError. É o esperado.
"""
import math

import numpy as np
import pandas as pd

from bayes import (COLUNAS, ajustar_gaussiana, br, carregar_compras, em_log, faixas,
                   gaussiana_log, gaussiana_prob, pct, resumo_junho, separar_compras, _nome_faixa)


def categorica_log(treino, y, novos, alpha=1.0):
    """log P(categoria | classe) de cada linha de novos, para cada classe.

    treino: Series com a categoria de cada caso conhecido.
    y: a classe de cada caso conhecido (mesmo tamanho de treino).
    novos: Series com a categoria de cada caso novo.
    Retorna um DataFrame com o índice de novos e uma coluna por classe, em ordem crescente:
    log((casos da classe com a categoria + alpha) ÷ (casos da classe + alpha × categorias do treino)).
    As categorias do treino são as que aparecem em treino, juntando todas as classes.
    Para com ValueError se alpha não for maior que zero, se treino e y tiverem tamanhos
    diferentes ou se novos tiver uma categoria que treino não tem. Não muda o que recebeu.
    """
    raise NotImplementedError("escreva categorica_log(treino, y, novos, alpha)")


# ----------------------------------------------------------------------------- o que roda junho
# Daqui para baixo, nada para mudar.

def _prob_fraude(logs):
    mx = logs.max(axis=1)
    e = np.exp(logs.sub(mx, axis=0))
    return e[1] / e.sum(axis=1)


def _imprimir_faixas(p, y):
    print("  P(fraude) que o modelo deu     compras   fraudes")
    for a, b, n, f in faixas(p, y):
        print(f"  {_nome_faixa(a, b):<28}{n:>9}{f:>10}")


def main():
    compras = carregar_compras()
    conh, novas = separar_compras(compras)
    y, yn = conh["fraude"], novas["fraude"]
    total_f = int(yn.sum())
    print(f"{len(conh)} compras conhecidas (janeiro a maio) e {len(novas)} compras de junho")

    print("\nA. As tentativas do cartão: a categoria e a curva normal")
    cat = categorica_log(conh["tentativas_cartao"], y, pd.Series(range(1, 7)))
    print("  tentativas   categoria: legítima    fraude   curva normal: legítima    fraude")
    curva = {}
    for k in (0, 1):
        parte = conh.loc[y == k, "tentativas_cartao"].astype(float)
        mu, dp = float(parte.mean()), float(parte.std(ddof=0))
        curva[k] = lambda a, b, mu=mu, dp=dp: (0.5 * (math.erf((b - mu) / (dp * math.sqrt(2))) -
                                                     math.erf((a - mu) / (dp * math.sqrt(2)))))
    for t in range(1, 7):
        c0, c1 = math.exp(cat.loc[t - 1, 0]), math.exp(cat.loc[t - 1, 1])
        print(f"  {t:>10}{pct(c0, 2):>21}{pct(c1, 2):>10}{pct(curva[0](t - 0.5, t + 0.5), 2):>24}"
              f"{pct(curva[1](t - 0.5, t + 0.5), 2):>10}")
    fora = {k: 1 - curva[k](0.5, 6.5) for k in (0, 1)}
    print(f"  {'fora de 1 a 6':>10}{'0,00%':>21}{'0,00%':>10}{pct(fora[0], 2):>24}{pct(fora[1], 2):>10}")

    print("\nB. Junho: as tentativas na curva normal ou como categoria")
    x, xn = em_log(conh), em_log(novas)
    m5 = ajustar_gaussiana(x, y)
    p5 = gaussiana_prob(xn, m5)
    f5, l5 = resumo_junho(p5, yn)
    print(f"  curva normal nas 5 colunas: marca {f5} das {total_f} fraudes e {l5} compras legítimas")
    _imprimir_faixas(p5, yn)
    quatro = [c for c in COLUNAS if c != "tentativas_cartao"]
    m4 = ajustar_gaussiana(x[quatro], y)
    logs = gaussiana_log(xn[quatro], m4) + categorica_log(conh["tentativas_cartao"], y, novas["tentativas_cartao"])
    pc = _prob_fraude(logs)
    fc, lc = resumo_junho(pc, yn)
    print(f"  curva normal em 4 colunas e tentativas como categoria: marca {fc} das {total_f} fraudes e {lc} compras legítimas")
    _imprimir_faixas(pc, yn)

    print("\nC. A coluna da Renata: a idade da conta de novo, vinda do cadastro")
    x2, xn2 = x.copy(), xn.copy()
    x2["dias_desde_cadastro"] = np.log1p(conh["idade_conta_dias"].astype(float))
    xn2["dias_desde_cadastro"] = np.log1p(novas["idade_conta_dias"].astype(float))
    m6 = ajustar_gaussiana(x2, y)
    p6 = gaussiana_prob(xn2, m6)
    f6, l6 = resumo_junho(p6, yn)
    print(f"  curva normal nas 5 colunas mais a cópia: marca {f6} das {total_f} fraudes e {l6} compras legítimas")
    _imprimir_faixas(p6, yn)


if __name__ == "__main__":
    main()
