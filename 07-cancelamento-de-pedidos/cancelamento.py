"""As contas da aula «Cancelamento de pedidos» do curso ML supervisionado.

    python3 cancelamento.py

Lê ../dados/loja/pedidos_export_completo.csv, a base compartilhada do kit. A preparação é a do
curso ML: preparação e validação, o pipeline_corrigido.py sem nenhuma mudança: treino de janeiro a
março, validação em abril e maio, junho fechado. O modelo é a regressão logística do treino.py,
pelo gradiente da log-loss. O classificador.py é o classificador pronto do curso anterior, usado
só para comparar. A parte 8 refaz a preparação do jeito do Rafael, para mostrar o que muda.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("../dados/loja/pedidos_export_completo.csv")
CORTES = (0.5, 1 / 3)
PASSOS = (0, 1, 10, 100, 1000, 5000)


def num(x, casas=1):
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x):
    return num(100 * x) + "%"


def matriz(y, ligar):
    y = np.asarray(y).astype(int)
    ligar = np.asarray(ligar).astype(bool)
    return {"vp": int((ligar & (y == 1)).sum()), "fp": int((ligar & (y == 0)).sum()),
            "fn": int((~ligar & (y == 1)).sum()), "vn": int((~ligar & (y == 0)).sum())}


def mostrar_matriz(m):
    n = sum(m.values())
    print("                        cancelou   não cancelou")
    print(f"   liga (nota acima)    VP {m['vp']:>4}        FP {m['fp']:>4}")
    print(f"   não liga             FN {m['fn']:>4}        VN {m['vn']:>4}")
    print(f"   ligações: {m['vp'] + m['fp']} | acertos (VP + VN): {m['vp'] + m['vn']} de {n}, "
          f"acurácia {pct((m['vp'] + m['vn']) / n)}")


def preparacao_do_rafael(base):
    """O atalho do Rafael: get_dummies em todas as colunas e escala com janeiro a maio juntos."""
    p = base[base["data_pedido"] <= pd.Timestamp("2026-05-31")]
    colunas = [c for c in p.columns if c not in ("id_pedido", "data_pedido", "cancelado")]
    x = pd.get_dummies(p[colunas], dtype=float)
    numericas = [c for c in x.columns if x[c].nunique() > 2]
    x[numericas] = (x[numericas] - x[numericas].mean()) / x[numericas].std(ddof=0)
    treino = p["data_pedido"] <= pd.Timestamp("2026-03-31")
    return x[treino], p["cancelado"][treino], x[~treino], p["cancelado"][~treino], colunas


def nome_curto(coluna):
    for prefixo in ("forma_pagamento_", "categoria_", "canal_"):
        if coluna.startswith(prefixo):
            return coluna[len(prefixo):]
    return coluna


def main():
    if not BASE.is_file():
        sys.exit(f"não achei {BASE}. Rode de dentro da pasta 07-cancelamento-de-pedidos do kit.")
    import classificador
    import pipeline_corrigido
    import treino

    base = pd.read_csv(BASE, parse_dates=["data_pedido"])
    x_tr, y_tr, x_va, y_va = pipeline_corrigido.preparar(base)
    params = pipeline_corrigido.ajustar(base[base["data_pedido"] <= pd.Timestamp(pipeline_corrigido.FIM_TREINO)])
    n_num = len(pipeline_corrigido.NUMERICAS)

    print("1. A preparação do curso anterior, sem mudar nada (pipeline_corrigido.py)")
    print(f"   treino: {len(y_tr)} pedidos de janeiro a março, {int(y_tr.sum())} cancelados ({pct(y_tr.mean())})")
    print(f"   validação: {len(y_va)} pedidos de abril e maio, {int(y_va.sum())} cancelados ({pct(y_va.mean())})")
    print("   junho, o teste, sai na leitura e não entra em nenhuma conta")
    print(f"   {x_tr.shape[1]} colunas: {n_num} números na escala do treino "
          f"e {x_tr.shape[1] - n_num} colunas de 0 e 1 das categorias")
    print(f"   ficam fora, porque nascem depois do pedido: {', '.join(pipeline_corrigido.NASCEM_DEPOIS)}")

    pronto = classificador.treinar(x_tr, y_tr)
    nota_pronto = classificador.nota(pronto, x_va)
    nomes = ["intercepto"] + list(x_tr.columns)
    w_pronto = pd.Series(pronto["pesos"], index=nomes)

    print(f"\n2. O treino pelo gradiente da log-loss: taxa {num(treino.TAXA)}, saindo de tudo zero")
    print("   comparado com o classificador pronto do curso anterior (maior diferença)")
    print("   passos   log-loss do treino   nota na validação   coeficiente   boleto contra pix")
    for passos in PASSOS:
        m = treino.treinar(x_tr, y_tr, passos=passos)
        wm = pd.Series(m["pesos"], index=nomes)
        perda = treino.log_loss(treino.nota(m, x_tr), y_tr) if passos else treino.log_loss(np.full(len(y_tr), 0.5), y_tr)
        dif_nota = float((treino.nota(m, x_va) - nota_pronto).abs().max())
        bp = (wm["forma_pagamento_boleto"] - wm["forma_pagamento_pix"]) - (
            w_pronto["forma_pagamento_boleto"] - w_pronto["forma_pagamento_pix"])
        print(f"   {passos:>6}   {num(perda, 4):>18}   {num(dif_nota, 4):>17}   {num(float((wm - w_pronto).abs().max()), 4):>11}"
              f"   {num(abs(bp), 4):>17}")
    print("   o classificador pronto: log-loss do treino "
          f"{num(treino.log_loss(classificador.nota(pronto, x_tr), y_tr), 4)}")
    modelo = m                                          # o último, de 5000 passos
    nota = treino.nota(modelo, x_va)
    w = pd.Series(modelo["pesos"], index=nomes)

    print("\n3. O modelo: score = intercepto + soma de coeficiente × coluna")
    print("   coluna                    coeficiente   × nas odds   um desvio vale")
    print(f"   {'intercepto':<25} {num(w['intercepto'], 3):>11}")
    for c in modelo["colunas"]:
        desvio = num(params["desvio"][c], 2) if c in pipeline_corrigido.NUMERICAS else ""
        print(f"   {c:<25} {num(w[c], 3):>11} {num(np.exp(w[c]), 2):>12}   {desvio:>14}".rstrip())
    print(f"   os {n_num} números estão na escala do treino: o coeficiente é de um desvio a mais")

    print("\n4. As categorias, uma contra a outra, com o resto igual")
    for a, b in [("forma_pagamento_boleto", "forma_pagamento_pix"), ("forma_pagamento_boleto", "forma_pagamento_cartao"),
                 ("categoria_Esporte", "categoria_Papelaria"), ("canal_app", "canal_site")]:
        d = w[a] - w[b]
        print(f"   {nome_curto(a)} contra {nome_curto(b)}: {num(d, 3)}, odds × {num(np.exp(d), 2)}")
    print("   boleto contra pix no classificador pronto: "
          f"{num(w_pronto['forma_pagamento_boleto'] - w_pronto['forma_pagamento_pix'], 3)}")

    print(f"\n5. A ordem: acerto do par na validação {pct(classificador.acerto_do_par(nota, y_va))}")
    nunca = matriz(y_va, np.zeros(len(y_va), dtype=bool))
    print(f"   nunca ligar acerta {nunca['vn']} de {len(y_va)}: acurácia {pct(nunca['vn'] / len(y_va))}")

    for i, corte in enumerate(CORTES, start=6):
        print(f"\n{i}. A matriz de confusão na validação, corte {num(corte, 3)}: liga quando a nota passa do corte")
        mostrar_matriz(matriz(y_va, nota > corte))

    xr_tr, yr_tr, xr_va, yr_va, colunas = preparacao_do_rafael(base)
    rafael = classificador.treinar(xr_tr, yr_tr)
    nota_r = classificador.nota(rafael, xr_va)
    print("\n8. A preparação refeita do Rafael: get_dummies em todas as colunas, escala com janeiro a maio")
    print(f"   colunas de entrada: {', '.join(colunas)}")
    print(f"   acerto do par na validação: {pct(classificador.acerto_do_par(nota_r, yr_va))}")
    print(f"   corte {num(0.5, 3)}:")
    mostrar_matriz(matriz(yr_va, nota_r > 0.5))


if __name__ == "__main__":
    main()
