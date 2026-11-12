"""O exercício da aula «Métricas e prevalência»: dois modelos de fraude nas compras de abril e maio.

    python3 comparar.py

O modelo da aula lê a hora como número, de 0 a 23. O modelo da Bruna troca a hora por uma
coluna de 0 e 1: madrugada, das 23h às 5h59. As outras colunas são as mesmas.
Junho fica fechado. A saída sai igual toda vez: não tem sorteio.
"""
import modelo as m

LISTAS = [27, 54, 100]      # quantas compras a equipe revisa em abril e maio


def main():
    treino, valid = m.compras()
    y = valid["fraude"].to_numpy()
    colunas_bruna = ["madrugada" if c == "hora" else c for c in m.COLUNAS_FRAUDE]
    modelos = [("modelo da aula", m.modelo_fraude(treino, valid, m.COLUNAS_FRAUDE)),
               ("modelo da Bruna", m.modelo_fraude(treino, valid, colunas_bruna))]

    print(f"abril e maio: {len(y)} compras, {y.sum()} fraudes ({m.pct(y.mean(), 2)})")
    print(f"modelo da aula:  {', '.join(m.COLUNAS_FRAUDE)}")
    print(f"modelo da Bruna: {', '.join(colunas_bruna)}")
    print()
    print("1. A lista inteira")
    print("                                       " + "".join(f"{nome:>17}" for nome, _ in modelos))
    print("acerto do par (área sob a curva ROC)   "
          + "".join(f"{m.pct(m.acerto_do_par(n, y), 2):>17}" for _, n in modelos))
    print("KS                                     "
          + "".join(f"{m.num(100 * m.ks(n, y)[0]):>17}" for _, n in modelos))
    print("precisão média (área sob a curva PR)   "
          + "".join(f"{m.pct(m.precisao_media(n, y)):>17}" for _, n in modelos))
    print()
    print("2. As listas de revisão: as maiores notas")
    print("revisadas  modelo            fraudes  sem fraude  fraudes fora  precisão  revocação     F1")
    for k in LISTAS:
        for nome, n in modelos:
            vp, fp, fn, vn = m.topo(n, y, k)
            p, r, f, _ = m.no_corte(vp, fp, fn, vn)
            print(f"{k:>9}  {nome:<16} {vp:>8} {fp:>11} {fn:>13} {m.pct(p):>9} {m.pct(r):>10} {m.pct(f):>6}")


if __name__ == "__main__":
    main()
