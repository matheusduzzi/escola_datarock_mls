"""Exercício da aula «Cancelamento de pedidos»: o modelo guardado e a matriz por forma de pagamento.

Escreva as quatro funções marcadas com NotImplementedError. Depois rode, de dentro da pasta da aula:

    python3 testar_meu_modelo.py
    python3 meu_modelo.py

Enquanto as funções não existem, os dois comandos param com NotImplementedError.
A preparação é o pipeline_corrigido.py do curso anterior e o treino é o treino.py da aula.
Não mexa nos dois: o exercício é usar o que já existe.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("../dados/loja/pedidos_export_completo.csv")
ARQUIVO_MODELO = Path("modelo.json")
CORTE = 1 / 3          # o corte do empate do curso anterior: R$ 5 da ligação ÷ R$ 15 do cancelamento


def matriz(y, ligar):
    """Retorna {"vp", "fp", "fn", "vn"}, com números inteiros.

    y: 1 quando o pedido cancelou, 0 quando foi entregue. ligar: 1 (ou True) quando a regra liga.
    VP: liga e cancelou. FP: liga e não cancelou. FN: não liga e cancelou. VN: não liga e não cancelou.
    Pare com ValueError se y e ligar tiverem tamanhos diferentes ou se tiverem algo além de 0 e 1.
    """
    raise NotImplementedError("escreva matriz(y, ligar)")


def por_grupo(y, ligar, grupo):
    """Retorna {valor do grupo: matriz dos pedidos desse grupo}, com os grupos em ordem alfabética.

    Pare com ValueError se grupo não tiver o mesmo tamanho de y.
    """
    raise NotImplementedError("escreva por_grupo(y, ligar, grupo)")


def salvar(modelo, caminho):
    """Grava o modelo em JSON: {"colunas": [...], "pesos": [intercepto, um por coluna]}.

    Pare com ValueError, sem gravar nada, se o número de pesos não for o de colunas mais um
    ou se algum peso não for um número (nan ou infinito).
    """
    raise NotImplementedError("escreva salvar(modelo, caminho)")


def carregar(caminho):
    """Lê o JSON do salvar e retorna {"colunas": lista, "pesos": array do NumPy}.

    Pare com ValueError se faltar "colunas" ou "pesos", ou se os dois não baterem como no salvar.
    """
    raise NotImplementedError("escreva carregar(caminho)")


def main():
    if not BASE.is_file():
        sys.exit(f"não achei {BASE}. Rode de dentro da pasta 07-cancelamento-de-pedidos do kit.")
    import pipeline_corrigido
    import treino

    base = pd.read_csv(BASE, parse_dates=["data_pedido"])
    x_tr, y_tr, x_va, y_va = pipeline_corrigido.preparar(base)
    modelo = treino.treinar(x_tr, y_tr)
    salvar(modelo, ARQUIVO_MODELO)
    guardado = carregar(ARQUIVO_MODELO)
    nota = treino.nota(guardado, x_va)
    dif = float((nota - treino.nota(modelo, x_va)).abs().max())
    print(f"modelo guardado em {ARQUIVO_MODELO}: {len(guardado['colunas'])} colunas e {len(guardado['pesos'])} pesos")
    print(f"maior diferença de nota entre o guardado e o recém-treinado, na validação: {dif:.6f}".replace(".", ","))

    ligar = (nota > CORTE).to_numpy()
    y = y_va.to_numpy()
    total = matriz(y, ligar)
    print(f"\nvalidação, abril e maio: {len(y)} pedidos; corte {CORTE:.3f}".replace(".", ","))
    print(f"matriz: VP {total['vp']}  FP {total['fp']}  FN {total['fn']}  VN {total['vn']}")

    forma = base.set_index("id_pedido").loc[y_va.index, "forma_pagamento"].to_numpy()
    grupos = por_grupo(y, ligar, forma)
    print("\npor forma de pagamento")
    print(f"{'forma':<8} {'pedidos':>7} {'cancelaram':>10} {'VP':>4} {'FP':>4} {'FN':>4} {'VN':>4} {'ligações':>8}")
    for g, m in grupos.items():
        print(f"{g:<8} {sum(m.values()):>7} {m['vp'] + m['fn']:>10} {m['vp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"{m['vn']:>4} {m['vp'] + m['fp']:>8}")
    soma = {k: sum(m[k] for m in grupos.values()) for k in total}
    print(f"\nos grupos somam a matriz total: {'sim' if soma == total else 'não'}")


if __name__ == "__main__":
    main()
