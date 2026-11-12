"""Aula «Calibração e limiar»: ordem, probabilidade e decisão no modelo de cancelamento.

Rodar de dentro da pasta da aula:
    python3 calibracao.py              # as seis partes da saída
    python3 calibracao.py --grafico    # também salva curva-calibracao.png (precisa do matplotlib)

Dois modelos treinados nos mesmos pedidos de janeiro a março: o sem peso e o com peso, em que
cada cancelado conta 4,04 vezes. Abril e maio medem. Junho fica fechado.
"""
import sys

import numpy as np

import modelo as m

BASE = "../dados/loja/pedidos_export_completo.csv"
PRECO_LIGACAO = 5
PRECO_CANCELAMENTO = 15
CORTE_EMPATE = PRECO_LIGACAO / PRECO_CANCELAMENTO
LIGACOES_POR_PEDIDO = 1 / 4
FAIXAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
FAIXAS_MAIO = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
PEDIDO_DO_CAIO = 1543


def faixa(nota, limites):
    """O número da faixa de cada nota. A última faixa inclui o 1."""
    return np.clip(np.searchsorted(limites, nota, side="right") - 1, 0, len(limites) - 2)


def custo(nota, y, corte):
    liga = nota >= corte
    certas = int((liga & (y == 1)).sum())
    passam = int((~liga & (y == 1)).sum())
    return int(liga.sum()), certas, passam, PRECO_LIGACAO * int(liga.sum()) + PRECO_CANCELAMENTO * passam


def brier(nota, y):
    return float(np.mean((nota - y) ** 2))


def linha_custo(nome, nota, y, corte, teto):
    lig, certas, passam, total = custo(nota, y, corte)
    cabe = "sim" if lig <= teto else "não"
    return f"{nome:<26}{lig:>9}{certas:>8}{passam:>8}{m.reais(total):>11}   {cabe}"


def main(grafico=False):
    base = m.ler(BASE)
    d = m.preparar(base)
    x_tr, y_tr, _ = d["treino"]
    x_ab, y_ab, id_ab = d["abril"]
    x_ma, y_ma, id_ma = d["maio"]
    x_va, y_va, id_va = np.r_[x_ab, x_ma], np.r_[y_ab, y_ma], np.r_[id_ab, id_ma]

    fator = (y_tr == 0).sum() / y_tr.sum()
    sem_peso = m.treinar(x_tr, y_tr)
    com_peso = m.treinar(x_tr, y_tr, peso_cancelado=fator)
    n0, n1 = m.nota(sem_peso, x_va), m.nota(com_peso, x_va)

    print(f"treino: {len(y_tr)} pedidos de janeiro a março, {y_tr.sum()} cancelados ({m.pct(y_tr.mean())})")
    print(f"abril: {len(y_ab)} pedidos, {y_ab.sum()} cancelados ({m.pct(y_ab.mean())}) | "
          f"maio: {len(y_ma)} pedidos, {y_ma.sum()} cancelados ({m.pct(y_ma.mean())})")
    print("junho, o teste, fica fechado: nenhuma conta usa esses pedidos")
    print(f"com peso: cada cancelado conta {m.num(fator, 2)} vezes no treino")
    i = int(np.where(id_va == PEDIDO_DO_CAIO)[0][0])
    print(f"pedido {PEDIDO_DO_CAIO}: nota {m.pct(n0[i])} sem peso, {m.pct(n1[i])} com peso")

    # 1. ordem
    teto_va = int(len(y_va) * LIGACOES_POR_PEDIDO)
    print(f"\n1. A ordem: abril e maio, {len(y_va)} pedidos, {y_va.sum()} cancelados")
    print(f"modelo      acerto do par   cancelaram entre as {teto_va} maiores notas")
    topo = {}
    for nome, nt in (("sem peso", n0), ("com peso", n1)):
        topo[nome] = set(np.argsort(-nt, kind="stable")[:teto_va])
        print(f"{nome:<12}{m.pct(m.acerto_do_par(nt, y_va)):>13}   {int(y_va[list(topo[nome])].sum()):>6} de {teto_va}")
    print(f"   pedidos que estão nas duas listas: {len(topo['sem peso'] & topo['com peso'])} de {teto_va}")
    print(f"   intercepto: sem peso {m.num(sem_peso[0], 3)}, com peso {m.num(com_peso[0], 3)}; "
          f"subiu {m.num(com_peso[0] - sem_peso[0], 3)} (o logaritmo de {m.num(fator, 2)} é {m.num(np.log(fator), 3)})")
    print(f"   maior diferença entre os outros {len(sem_peso) - 1} coeficientes: "
          f"{m.num(np.abs(com_peso[1:] - sem_peso[1:]).max(), 3)}")

    # 2. probabilidade
    print("\n2. A probabilidade: a curva de calibração em abril e maio")
    print("                       sem peso                          com peso")
    print("faixa da nota   pedidos  nota média  cancelaram   pedidos  nota média  cancelaram")
    f0, f1 = faixa(n0, FAIXAS), faixa(n1, FAIXAS)
    curva = []
    for k in range(len(FAIXAS) - 1):
        partes = []
        for nt, ff in ((n0, f0), (n1, f1)):
            s = ff == k
            partes.append((int(s.sum()), float(nt[s].mean()) if s.any() else None,
                           float(y_va[s].mean()) if s.any() else None))
        curva.append(partes)
        texto = f"{m.num(FAIXAS[k])} a {m.num(FAIXAS[k + 1])}  "
        for n, med, real in partes:
            texto += f"{n:>9}{m.pct(med):>12}{m.pct(real):>12}" if n else f"{n:>9}{'-':>12}{'-':>12}"
            texto += "  "
        print(texto.rstrip())
    print(f"soma das notas: sem peso {m.num(n0.sum())}, com peso {m.num(n1.sum())}; cancelaram {y_va.sum()}")
    constante = np.full(len(y_va), y_va.mean())
    print(f"Brier: sem peso {m.num(brier(n0, y_va), 3)}; com peso {m.num(brier(n1, y_va), 3)}; "
          f"a mesma nota de {m.pct(y_va.mean())} para todos {m.num(brier(constante, y_va), 3)}")

    # 3. decisão
    print(f"\n3. A decisão: liga quando a nota passa do corte do empate, {m.num(CORTE_EMPATE, 3)}")
    print(f"preços: ligação {m.reais(PRECO_LIGACAO)}, cancelamento que passa {m.reais(PRECO_CANCELAMENTO)}; "
          f"teto: 1 ligação para cada 4 pedidos ({teto_va} aqui)")
    print(f"{'regra':<26}{'ligações':>9}{'certas':>8}{'passam':>8}{'custo':>11}   cabe?")
    print(linha_custo("nunca liga", n0, y_va, 2.0, teto_va))
    print(linha_custo("sem peso, corte 0,333", n0, y_va, CORTE_EMPATE, teto_va))
    print(linha_custo("com peso, corte 0,333", n1, y_va, CORTE_EMPATE, teto_va))
    for nome, nt in (("sem peso", n0), ("com peso", n1)):
        print(linha_custo(f"{nome}, {teto_va} maiores notas", nt, y_va,
                          float(np.sort(nt)[::-1][teto_va - 1]), teto_va))

    # 4. corrigir a escala
    a_ab, a_ma = m.nota(com_peso, x_ab), m.nota(com_peso, x_ma)
    escala = m.ajustar_escala(a_ab, y_ab)
    corrigida = m.corrigir(escala, a_ma)
    equivalente = m.corte_equivalente(escala, CORTE_EMPATE)
    print("\n4. Corrigir a escala do modelo com peso: ajustada em abril, medida em maio")
    print(f"nota corrigida = 1 / (1 + e^-({m.num(escala[0], 3)} + {m.num(escala[1], 3)} × logit da nota))")
    print(f"o corte 0,333 na nota corrigida é o corte {m.num(equivalente, 3)} na nota sem correção")
    print(f"maio, {len(y_ma)} pedidos     acerto do par  nota média  soma das notas  cancelaram  Brier")
    for nome, nt in (("sem peso", m.nota(sem_peso, x_ma)), ("com peso, sem correção", a_ma),
                     ("com peso, corrigida", corrigida)):
        print(f"{nome:<24}{m.pct(m.acerto_do_par(nt, y_ma)):>14}{m.pct(nt.mean()):>12}{m.num(nt.sum()):>16}"
              f"{y_ma.sum():>12}{m.num(brier(nt, y_ma), 3):>7}")
    print("                    com peso, sem correção            com peso, corrigida")
    print("faixa da nota   pedidos  nota média  cancelaram   pedidos  nota média  cancelaram")
    g0, g1 = faixa(a_ma, FAIXAS_MAIO), faixa(corrigida, FAIXAS_MAIO)
    for k in range(len(FAIXAS_MAIO) - 1):
        texto = f"{m.num(FAIXAS_MAIO[k])} a {m.num(FAIXAS_MAIO[k + 1])}  "
        for nt, ff in ((a_ma, g0), (corrigida, g1)):
            s = ff == k
            texto += (f"{int(s.sum()):>9}{m.pct(nt[s].mean()):>12}{m.pct(y_ma[s].mean()):>12}  " if s.any()
                      else f"{0:>9}{'-':>12}{'-':>12}  ")
        print(texto.rstrip())

    # 5. tabela de custos em maio
    teto_ma = int(len(y_ma) * LIGACOES_POR_PEDIDO)
    print(f"\n5. A tabela de custos em maio, com a nota corrigida (teto: {teto_ma} ligações)")
    print(f"{'corte':<26}{'ligações':>9}{'certas':>8}{'passam':>8}{'custo':>11}   cabe?")
    print(linha_custo("nunca liga", corrigida, y_ma, 2.0, teto_ma))
    for corte in (0.2, 0.25, 0.3, CORTE_EMPATE, 0.4, 0.5, 0.6):
        rotulo = m.num(corte, 3) + ("  <- o empate" if corte == CORTE_EMPATE else "")
        print(linha_custo(rotulo, corrigida, y_ma, corte, teto_ma))
    print(linha_custo("sem peso, corte 0,333", m.nota(sem_peso, x_ma), y_ma, CORTE_EMPATE, teto_ma))

    s = (corrigida >= CORTE_EMPATE) & (corrigida < 0.4)
    print(f"   entre 0,333 e 0,40: {int(s.sum())} pedidos, nota média {m.pct(corrigida[s].mean())}, "
          f"{m.num(corrigida[s].sum())} cancelamentos esperados, {int(y_ma[s].sum())} cancelaram")

    if grafico:
        desenhar(curva)


def desenhar(curva):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="nota = o que aconteceu")
    for j, (nome, cor) in enumerate((("sem peso", "#52645C"), ("com peso", "#C0504D"))):
        pontos = [(p[j][1], p[j][2]) for p in curva if p[j][0]]
        ax.plot(*zip(*pontos), marker="o", color=cor, label=nome)
    ax.set_xlabel("nota média da faixa")
    ax.set_ylabel("parte que cancelou")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    virgula = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax.xaxis.set_major_formatter(virgula)
    ax.yaxis.set_major_formatter(virgula)
    ax.set_title("Curva de calibração, abril e maio")
    ax.legend()
    fig.savefig("curva-calibracao.png", dpi=120, bbox_inches="tight")
    print("\ncurva-calibracao.png salva")


if __name__ == "__main__":
    main(grafico="--grafico" in sys.argv[1:])
