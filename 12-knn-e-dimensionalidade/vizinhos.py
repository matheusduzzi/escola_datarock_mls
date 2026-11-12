"""KNN nas compras com cartão da Loja Exemplo, só com o pandas e o NumPy.

Rode de dentro da pasta da aula:  python3 vizinhos.py

Também serve de módulo para o exercício:

    from vizinhos import COLUNAS, br, carregar, separar

As compras de janeiro a maio são as que o KNN já conhece (com o rótulo de fraude). As de
junho fazem o papel de compras novas: o programa procura os vizinhos de cada uma entre as de
janeiro a maio e vê se a maioria deles foi fraude.

As colunas de ruído da parte 5 são números aleatórios gerados com um gerador fixo, então a
saída sai igual toda vez.
"""
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
BASE_PADRAO = AQUI.parent / "dados" / "fraude" / "transacoes.csv"
COLUNAS = ["valor", "idade_conta_dias", "tentativas_cartao", "distancia_entrega_km", "hora"]
CURTO = {"valor": "valor", "idade_conta_dias": "conta", "tentativas_cartao": "tent",
         "distancia_entrega_km": "dist", "hora": "hora"}
NOVA = 6912
MAO = ["valor", "tentativas_cartao"]
CANDIDATAS = [5350, 4188, 1055, 6626, 2662, 1047]
GERADOR_RUIDO = 2026


def carregar(caminho=BASE_PADRAO):
    """Lê as compras e confere as colunas que as aulas usam."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com dados/ ao lado.")
    dados = pd.read_csv(caminho)
    faltam = [c for c in COLUNAS + ["id_transacao", "data_hora", "fraude"] if c not in dados.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    if dados[COLUNAS + ["fraude"]].isna().any().any():
        raise ValueError("há valores vazios nas colunas usadas")
    return dados.set_index("id_transacao", drop=False)


def separar(dados):
    """Janeiro a maio (o que o KNN conhece) e junho (as compras novas)."""
    junho = dados["data_hora"].str.startswith("2026-06-")
    return dados[~junho], dados[junho]


def br(x, casas=2):
    """Número com vírgula decimal e ponto de milhar, como na aula."""
    if round(x, casas) == 0:
        x = 0.0
    s = f"{x:,.{casas}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def distancias(conhecidas, nova, desvio):
    """Distância (Pitágoras) da compra nova a cada compra conhecida, com cada diferença dividida pelo desvio."""
    dif = (conhecidas.to_numpy(float) - nova.to_numpy(float)) / desvio.to_numpy(float)
    return np.sqrt((dif ** 2).sum(axis=1))


def votar(dist, rotulos, k):
    """Os k mais perto (empate de distância: vale a ordem das compras) e quantos deles foram fraude."""
    ordem = np.argsort(dist, kind="stable")[:k]
    return ordem, int(rotulos[ordem].sum())


def junho(conh, novas, rot_conh, rot_novas, desvio, ks):
    """Para cada k: [fraudes de junho que o KNN marca, compras legítimas que ele marca].

    Dividir pelo desvio sem tirar a média dá as mesmas distâncias: a média some na diferença.
    """
    x = conh.to_numpy(float) / desvio
    res = {k: [0, 0] for k in ks}
    for linha, fraude in zip(novas.to_numpy(float) / desvio, rot_novas):
        ordem = np.argsort(np.sqrt(((x - linha) ** 2).sum(axis=1)), kind="stable")
        for k in ks:
            if rot_conh[ordem[:k]].sum() * 2 > k:
                res[k][0 if fraude else 1] += 1
    return res


def legitimas(n):
    return f"{n} compra legítima" if n == 1 else f"{n} compras legítimas"


def main():
    dados = carregar()
    conh, novas = separar(dados)
    rc, rn = conh["fraude"].to_numpy(), novas["fraude"].to_numpy()
    print(f"{len(conh)} compras de {conh['data_hora'].min()[:10]} a {conh['data_hora'].max()[:10]} "
          f"({rc.sum()} fraudes) e {len(novas)} de junho ({rn.sum()} fraudes)")
    print("colunas: " + ", ".join(COLUNAS))
    desvio = conh[COLUNAS].std(ddof=0)

    # 1. À mão
    nova = dados.loc[NOVA]
    print()
    print(f"1. À mão: a compra {NOVA} e seis compras de janeiro a maio, com duas colunas")
    print(f"  compra {NOVA}: valor {br(nova['valor'])}, tentativas {nova['tentativas_cartao']}")
    print(f"  desvio padrão de janeiro a maio: valor {br(desvio['valor'])}, "
          f"tentativas {br(desvio['tentativas_cartao'], 4)}")
    print("  compra      valor  tent  fraude | dif valor  dif tent | sem padronizar  padronizada")
    cand = dados.loc[CANDIDATAS]
    cru = distancias(cand[MAO], nova[MAO], pd.Series(1.0, index=MAO))
    pad = distancias(cand[MAO], nova[MAO], desvio[MAO])
    for (i, c), dc, dp in zip(cand.iterrows(), cru, pad):
        print(f"  {i:>6} {br(c['valor']):>10} {c['tentativas_cartao']:>5} {c['fraude']:>7} |"
              f" {br(c['valor'] - nova['valor']):>9} {c['tentativas_cartao'] - nova['tentativas_cartao']:>9} |"
              f" {br(dc):>14} {br(dp):>12}")
    rcand = cand["fraude"].to_numpy()
    for nome, dd in (("sem padronizar:", cru), ("padronizada:", pad)):
        ordem, votos = votar(dd, rcand, 3)
        ids = ", ".join(str(CANDIDATAS[j]) for j in ordem)
        print(f"  K = 3 {nome:<15} {ids} -> {votos} de 3 fraudes: {'fraude' if votos * 2 > 3 else 'legítima'}")

    # 2. Cinco colunas, contra todas as compras de janeiro a maio
    print()
    print(f"2. A compra {NOVA} com as cinco colunas, contra as {len(conh)} compras de janeiro a maio")
    print("  compra  " + "".join(f"{CURTO[c]:>10}" for c in COLUNAS) + f"{'fraude':>8}")
    print(f"  {NOVA:>6}  " + "".join(f"{br(nova[c], 2 if c == 'valor' else 0 if c != 'distancia_entrega_km' else 1):>10}"
                                     for c in COLUNAS) + f"{nova['fraude']:>8}")
    um = pd.Series(1.0, index=COLUNAS)
    for nome, dv in (("sem padronizar", um), ("com as colunas padronizadas", desvio)):
        dd = distancias(conh[COLUNAS], nova[COLUNAS], dv)
        ordem, votos = votar(dd, rc, 5)
        print(f"  os 5 vizinhos {nome}:")
        partes = np.zeros(len(COLUNAS))
        for j in ordem:
            c = conh.iloc[j]
            partes += ((c[COLUNAS].to_numpy(float) - nova[COLUNAS].to_numpy(float)) / dv.to_numpy(float)) ** 2
            print(f"  {c['id_transacao']:>6}  " + "".join(
                f"{br(c[col], 2 if col == 'valor' else 0 if col != 'distancia_entrega_km' else 1):>10}"
                for col in COLUNAS) + f"{c['fraude']:>8}   distância {br(dd[j])}")
        print(f"    {votos} de 5 fraudes. Parte de cada coluna na soma dos quadrados dos 5: "
              + ", ".join(f"{CURTO[c]} {br(100 * p / partes.sum(), 0)}%" for c, p in zip(COLUNAS, partes)))

    # 3. Junho inteiro, K = 5
    print()
    print(f"3. As {len(novas)} compras de junho, K = 5")
    for nome, dv in (("sem padronizar:", um), ("padronizada:", desvio)):
        res = junho(conh[COLUNAS], novas[COLUNAS], rc, rn, dv.to_numpy(float), [5])
        print(f"  {nome:<15} marca {res[5][0]} das {rn.sum()} fraudes e {legitimas(res[5][1])}")
    print(f"  cada compra nova mede {len(conh)} distâncias; junho inteiro, {br(len(conh) * len(novas), 0)}")

    # 4. O tamanho da vizinhança
    print()
    print("4. Padronizada, com K diferentes")
    ks = [1, 5, 15, 51, 101]
    res = junho(conh[COLUNAS], novas[COLUNAS], rc, rn, desvio.to_numpy(float), ks)
    print("  K       fraudes marcadas   legítimas marcadas   fraudes que precisa entre os vizinhos")
    for k in ks:
        print(f"  {k:<7} {res[k][0]:>10} de {rn.sum():<6} {res[k][1]:>14} {k // 2 + 1:>25}")

    # 5. Colunas de ruído
    print()
    print("5. Colunas de ruído (média 0 e desvio 1, sem relação com fraude), K = 5")
    print("  colunas no total   fraudes marcadas   legítimas marcadas   mais longe ÷ mais perto (mediana)")
    gerador = np.random.default_rng(GERADOR_RUIDO)
    base_c = ((conh[COLUNAS] - conh[COLUNAS].mean()) / desvio).to_numpy(float)
    base_n = ((novas[COLUNAS] - conh[COLUNAS].mean()) / desvio).to_numpy(float)
    lado = 3 ** 0.5
    for extra in (0, 5, 20, 50, 200):
        rc_extra = gerador.uniform(-lado, lado, (len(conh), extra))
        rn_extra = gerador.uniform(-lado, lado, (len(novas), extra))
        xc, xn = np.hstack([base_c, rc_extra]), np.hstack([base_n, rn_extra])
        marc, falsas, razoes = 0, 0, []
        for linha, fraude in zip(xn, rn):
            dd = np.sqrt(((xc - linha) ** 2).sum(axis=1))
            ordem = np.argsort(dd, kind="stable")
            if rc[ordem[:5]].sum() * 2 > 5:
                marc, falsas = marc + bool(fraude), falsas + (not fraude)
            razoes.append(dd[ordem[-1]] / dd[ordem[0]] if dd[ordem[0]] > 0 else np.inf)
        print(f"  {len(COLUNAS) + extra:<18} {marc:>9} de {rn.sum():<6} {falsas:>14} {br(float(np.median(razoes))):>22}")

    # 6. O cubo
    print()
    print("6. Para pegar 1% das compras em volta de um ponto, num cubo de lado 1")
    print("  colunas   lado do cubinho que pega 1% do volume")
    for d in (1, 2, 3, 5, 10, 20, 100):
        print(f"  {d:<9} {br(100 * 0.01 ** (1 / d), 1):>6}% do lado")


if __name__ == "__main__":
    main()
