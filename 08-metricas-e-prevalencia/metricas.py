"""As contas da aula «Métricas e prevalência».

    python3 metricas.py           # as seis partes da aula
    python3 metricas.py --curvas  # os pontos das curvas ROC e PR da fraude, um por linha

Junho fica fechado nas duas bases. A saída sai igual toda vez: não tem sorteio.
"""
import sys

import numpy as np

import modelo as m

PRECO_LIGACAO, PRECO_CANCELAMENTO = 5, 15
CORTES = [0.1, 0.2, 0.333, 0.5]
REVISA = 54          # a equipe da Bruna revisa 27 compras por mês: 54 em abril e maio


def linha_corte(rotulo, vp, fp, fn, vn):
    p, r, f1, fpr = m.no_corte(vp, fp, fn, vn)
    custo = PRECO_LIGACAO * (vp + fp) + PRECO_CANCELAMENTO * fn
    return (f"{rotulo:>6} {vp + fp:>9} {vp:>4} {fp:>4} {fn:>4} {vn:>4} {m.pct(p):>9} {m.pct(r):>10}"
            f" {m.pct(f1):>6} {m.pct(fpr):>15}   R$ {m.num(custo, 0)}")


def curvas(nota, y, nome):
    """Um ponto por fraude, da maior nota para a menor: o corte é a nota dessa fraude."""
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    for c in np.unique(nota[y == 1])[::-1]:          # um ponto a cada fraude que entra na lista
        vp, fp, fn, vn = m.matriz(nota, y, c)
        if vp + fp:
            print(f"{nome};{c:.6f};{vp / n1:.4f};{fp / n0:.4f};{vp / (vp + fp):.4f}")


def main():
    nota_c, y_c, y_treino = m.cancelamento()
    treino, valid = m.compras()
    y_f = valid["fraude"].to_numpy()
    nota_valor = valid["valor"].to_numpy()
    nota_f = m.modelo_fraude(treino, valid, m.COLUNAS_FRAUDE)

    if "--curvas" in sys.argv:
        print("modelo;corte;revocacao;falso_positivo;precisao")
        curvas(nota_valor, y_f, "regra do valor")
        curvas(nota_f, y_f, "modelo")
        return

    print("1. As duas bases, separadas pela data")
    print(f"cancelamento: treino {len(y_treino)} pedidos de janeiro a março, {y_treino.sum()} cancelados"
          f" ({m.pct(y_treino.mean())})")
    print(f"              abril e maio: {len(y_c)} pedidos, {y_c.sum()} cancelados ({m.pct(y_c.mean())})")
    print(f"fraude:       treino {len(treino)} compras de janeiro a março, {treino['fraude'].sum()} fraudes"
          f" ({m.pct(treino['fraude'].mean(), 2)})")
    print(f"              abril e maio: {len(y_f)} compras, {y_f.sum()} fraudes ({m.pct(y_f.mean(), 2)})")
    print("junho, o teste, fica fechado nas duas bases")

    print()
    print("2. Cancelamento: a matriz e as métricas em quatro cortes (liga quando a nota passa do corte)")
    print(" corte  ligações   VP   FP   FN   VN  precisão  revocação     F1  falso positivo   custo")
    for c in CORTES:
        print(linha_corte(m.num(c, 3), *m.matriz(nota_c, y_c, c)))
    f1, corte_f1 = max((m.no_corte(*m.matriz(nota_c, y_c, c))[2], c) for c in np.unique(nota_c))
    print(linha_corte(m.num(corte_f1, 3), *m.matriz(nota_c, y_c, corte_f1)))
    print(f"   a última linha é o corte do maior F1; o teto da equipe é de {len(y_c) // 4} ligações")

    print()
    print("3. Cancelamento: as métricas que olham a lista inteira")
    ks, onde = m.ks(nota_c, y_c)
    print(f"acerto do par (área sob a curva ROC)   {m.pct(m.acerto_do_par(nota_c, y_c))}")
    print(f"KS                                     {m.num(100 * ks)}, no corte {m.num(onde, 3)}")
    print(f"precisão média (área sob a curva PR)   {m.pct(m.precisao_media(nota_c, y_c))}")
    print(f"   a mesma nota para todos: acerto do par 50,0%, KS 0,0, precisão média {m.pct(y_c.mean())}")

    print()
    print("4. Fraude: a regra do valor contra o modelo, nas compras de abril e maio")
    print("                                       regra do valor     modelo")
    print(f"acerto do par (área sob a curva ROC)   {m.pct(m.acerto_do_par(nota_valor, y_f)):>14}"
          f" {m.pct(m.acerto_do_par(nota_f, y_f)):>10}")
    print(f"KS                                     {m.num(100 * m.ks(nota_valor, y_f)[0]):>14}"
          f" {m.num(100 * m.ks(nota_f, y_f)[0]):>10}")
    print(f"precisão média (área sob a curva PR)   {m.pct(m.precisao_media(nota_valor, y_f)):>14}"
          f" {m.pct(m.precisao_media(nota_f, y_f)):>10}")
    print(f"   a mesma nota para todos: acerto do par 50,0%, KS 0,0, precisão média {m.pct(y_f.mean(), 2)}")
    n1, n0 = int(y_f.sum()), int(len(y_f) - y_f.sum())
    errados = [round((1 - m.acerto_do_par(n, y_f)) * n1 * n0) for n in (nota_valor, nota_f)]
    print(f"   pares fraude contra compra sem fraude: {m.num(n1 * n0, 0)}; a fraude fica com a nota menor"
          f" em {m.num(errados[0], 0)} (regra do valor) e {m.num(errados[1], 0)} (modelo)")
    print(f"   nunca marcar fraude acerta {m.num(n0, 0)} de {m.num(len(y_f), 0)} compras: acurácia {m.pct(n0 / len(y_f))}")
    print(f"   modelo: logística com {', '.join(m.COLUNAS_FRAUDE)}, na escala do treino")

    print()
    print(f"5. Fraude: a lista que a equipe da Bruna revisa, as {REVISA} maiores notas de abril e maio")
    print("regra             revisadas  fraudes  sem fraude  fraudes fora  precisão  revocação     F1")
    for nome, nota in [("regra do valor", nota_valor), ("modelo", nota_f)]:
        vp, fp, fn, vn = m.topo(nota, y_f, REVISA)
        p, r, f, _ = m.no_corte(vp, fp, fn, vn)
        print(f"{nome:<17} {vp + fp:>9} {vp:>8} {fp:>11} {fn:>13} {m.pct(p):>9} {m.pct(r):>10} {m.pct(f):>6}")

    print()
    vp, fp, fn, vn = m.matriz(nota_c, y_c, 0.333)
    _, r, _, fpr = m.no_corte(vp, fp, fn, vn)
    print(f"6. A mesma regra em outra mistura: o cancelamento no corte 0,333, com revocação {m.pct(r)}"
          f" e falso positivo {m.pct(fpr)}")
    print("parte de positivos na base     precisão   marcados à toa para cada acerto")
    for rotulo, pi in [("20,5% (a do cancelamento)", y_c.mean()), ("1,19% (a da fraude)", y_f.mean())]:
        p = m.precisao_com_outra_mistura(r, fpr, pi)
        print(f"{rotulo:<29} {m.pct(p):>9}   {m.num((1 - p) / p)}")
    print("   a revocação e o falso positivo não mudam: cada um olha uma classe só")


if __name__ == "__main__":
    main()
