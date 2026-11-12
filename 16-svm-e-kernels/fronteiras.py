"""SVM e kernels nos pedidos da Loja Exemplo, só com o pandas e o NumPy.

Rode de dentro da pasta da aula:  python3 fronteiras.py

Também serve de módulo para o exercício:

    from fronteiras import br, carregar_entregas, k_linear, k_quadratico, treinar

Duas bases:
- dados/loja/pedidos.csv, na raiz do kit: os pedidos com o rótulo de cancelamento. A aula usa
  dez pedidos de janeiro a março, com duas colunas, para a margem à mão.
- entregas.csv, nessa pasta: os pedidos entregues de motoboy a partir do centro de distribuição,
  com o endereço em km (leste_km, norte_km). Março é o treino; abril faz o papel de pedidos novos.

O SVM é treinado aqui mesmo, pela função treinar(): ela resolve o problema dual mexendo em dois
alfas por vez (o método SMO, o mesmo tipo usado pela libsvm). Não há sorteio: a saída sai igual
toda vez.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
PEDIDOS = AQUI.parent / "dados" / "loja" / "pedidos.csv"
ENTREGAS = AQUI / "entregas.csv"
DUAS = ["prazo_prometido_dias", "pedidos_anteriores"]
MAPA = ["leste_km", "norte_km"]
DEZ = [1044, 36, 15, 1546, 1194, 587, 62, 317, 478, 808]
TEIMOSOS = [1207, 784]
PAR = (5013, 5024)


# ---------- leitura ----------

def carregar_pedidos(caminho=PEDIDOS):
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com dados/ na raiz do kit.")
    dados = pd.read_csv(caminho)
    faltam = [c for c in DUAS + ["id_pedido", "data_pedido", "cancelado"] if c not in dados.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    return dados.set_index("id_pedido", drop=False)


def carregar_entregas(caminho=ENTREGAS):
    """Março (treino) e abril (pedidos novos) do entregas.csv."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula.")
    dados = pd.read_csv(caminho)
    faltam = [c for c in MAPA + ["id_pedido", "data_pedido", "cancelado"] if c not in dados.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    dados = dados.set_index("id_pedido", drop=False)
    marco = dados["data_pedido"].str.startswith("2026-03")
    return dados[marco], dados[~marco]


def sinal(rotulo):
    """0 e 1 viram −1 e +1: o SVM trabalha com a classe dos dois lados do zero."""
    return np.where(np.asarray(rotulo) == 1, 1.0, -1.0)


def br(x, casas=2):
    """Número com vírgula decimal e ponto de milhar, como na aula."""
    if round(x, casas) == 0:
        x = 0.0
    s = f"{x:,.{casas}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


# ---------- kernels ----------

def _produto(A, B):
    """Produto escalar de cada linha de A com cada linha de B, sem a multiplicação de matrizes do NumPy."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return (A[:, None, :] * B[None, :, :]).sum(axis=2)


def k_linear(A, B):
    return _produto(A, B)


def k_quadratico(A, B):
    """(a · b)²: o mesmo que o produto escalar das colunas a², √2·a1·a2 e b²."""
    return _produto(A, B) ** 2


def k_rbf(A, B, gamma):
    """exp(−gamma · distância²): 1 no mesmo ponto, perto de 0 longe."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return np.exp(-gamma * ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


def phi(X):
    """As três colunas que o kernel quadrático usa sem criar: a², √2·a1·a2, b²."""
    X = np.asarray(X, float)
    return np.column_stack([X[:, 0] ** 2, math.sqrt(2) * X[:, 0] * X[:, 1], X[:, 1] ** 2])


# ---------- treino e decisão ----------

def treinar(X, y, kernel, C, tol=1e-6, max_iter=1_000_000):
    """SVM pelo problema dual: alfas entre 0 e C, soma de alfa·y igual a zero.

    A cada passo escolhe o par de alfas que mais viola as condições de ótimo (seleção de
    segunda ordem, Fan, Chen e Lin, 2005) e resolve esse par na mão. Para quando a maior
    violação fica abaixo de tol. Retorna um dicionário com X, y (−1 e +1), alfa, b, kernel e C.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    K = kernel(X, X)
    diag = np.diag(K).copy()
    a = np.zeros(n)
    u = np.zeros(n)  # u[k] = soma de alfa·y·K(x, x_k), sem o b
    zero = 1e-12 * C
    for passo in range(1, max_iter + 1):
        v = y - u
        sobe = ((a < C) & (y > 0)) | ((a > 0) & (y < 0))
        desce = ((a < C) & (y < 0)) | ((a > 0) & (y > 0))
        ids = np.where(sobe)[0]
        i = ids[np.argmax(v[ids])]
        idd = np.where(desce)[0]
        if v[i] - v[idd].min() < tol:
            break
        cand = idd[v[idd] < v[i]]
        eta = np.maximum(diag[i] + diag[cand] - 2 * K[i, cand], 1e-12)
        j = cand[np.argmin(-((v[i] - v[cand]) ** 2) / eta)]
        e = max(K[i, i] + K[j, j] - 2 * K[i, j], 1e-12)
        if y[i] == y[j]:
            lo, hi = max(0.0, a[i] + a[j] - C), min(C, a[i] + a[j])
        else:
            lo, hi = max(0.0, a[j] - a[i]), min(C, C + a[j] - a[i])
        aj = min(max(a[j] + y[j] * (v[j] - v[i]) / e, lo), hi)
        ai = a[i] + y[i] * y[j] * (a[j] - aj)
        ai = 0.0 if ai < zero else (C if ai > C - zero else ai)
        aj = 0.0 if aj < zero else (C if aj > C - zero else aj)
        u += (ai - a[i]) * y[i] * K[i] + (aj - a[j]) * y[j] * K[j]
        a[i], a[j] = ai, aj
    else:
        raise RuntimeError(f"o treino não parou em {max_iter} passos")
    v = y - u
    livres = (a > 0) & (a < C)
    if livres.any():
        b = float(v[livres].mean())
    else:
        de_baixo = [v[t] for t in range(n) if (a[t] == 0 and y[t] > 0) or (a[t] == C and y[t] < 0)]
        de_cima = [v[t] for t in range(n) if (a[t] == 0 and y[t] < 0) or (a[t] == C and y[t] > 0)]
        b = (max(de_baixo) + min(de_cima)) / 2
    return {"X": X, "y": y, "alfa": a, "b": b, "kernel": kernel, "C": C, "passos": passo}


def decisao(modelo, X):
    """f(x) = soma de alfa·y·K(x_i, x) + b. Positivo: cancela. O tamanho diz a folga."""
    K = modelo["kernel"](modelo["X"], np.asarray(X, float))
    return ((modelo["alfa"] * modelo["y"])[:, None] * K).sum(axis=0) + modelo["b"]


def pesos(modelo):
    """w = soma de alfa·y·x, só para o kernel linear: f(x) = w · x + b."""
    return ((modelo["alfa"] * modelo["y"])[:, None] * modelo["X"]).sum(axis=0)


def cortes(f_da_linha, de=-12.0, ate=12.0):
    """Onde f troca de sinal na linha leste-oeste (norte = 0), com uma casa."""
    xs = np.linspace(de, ate, int(round((ate - de) * 100)) + 1)
    f = f_da_linha(np.column_stack([xs, np.zeros_like(xs)]))
    s = np.sign(f)
    achados = []
    for k in np.where(s[1:] != s[:-1])[0]:
        lo, hi = xs[k], xs[k + 1]
        for _ in range(40):
            meio = (lo + hi) / 2
            if np.sign(f_da_linha(np.array([[meio, 0.0]]))[0]) == s[k]:
                lo = meio
            else:
                hi = meio
        achados.append(round((lo + hi) / 2, 1))
    return achados


def texto_cortes(lista):
    return "nenhum" if not lista else ", ".join(br(c, 1) for c in lista)


def contar(f, y):
    """(cancelados marcados, entregues marcados, erros) com a regra f > 0 = cancela."""
    marca = f > 0
    cancelou = np.asarray(y) > 0
    return int((marca & cancelou).sum()), int((marca & ~cancelou).sum()), int((marca != cancelou).sum())


# ---------- a aula ----------

def parte1(ped):
    print("1. Os dez pedidos de janeiro a março e a reta de maior margem (C = 1000)")
    X, y = ped.loc[DEZ, DUAS].to_numpy(float), sinal(ped.loc[DEZ, "cancelado"])
    m = treinar(X, y, k_linear, 1000.0)
    w, f = pesos(m), decisao(m, X)
    norma = math.hypot(*w)
    print(f"  f = {br(w[0])} × prazo {'−' if w[1] < 0 else '+'} {br(abs(w[1]))} × anteriores "
          f"{'−' if m['b'] < 0 else '+'} {br(abs(m['b']))}")
    print(f"  largura da faixa = 2 ÷ {br(norma)} = {br(2 / norma)}")
    print("  pedido  prazo  anteriores  cancelou |      f   distância à reta   alfa")
    for k, pid in enumerate(DEZ):
        print(f"  {pid:>6}  {int(X[k, 0]):>5}  {int(X[k, 1]):>10}  {int(y[k] > 0):>8} | {br(f[k]):>6}"
              f"   {br(abs(f[k]) / norma):>16}   {br(m['alfa'][k]):>4}")
    sv = [DEZ[k] for k in np.where(m["alfa"] > 0)[0]]
    print(f"  vetores de suporte: {', '.join(map(str, sv))}")
    return m


def parte2(ped, m):
    print("\n2. Três retas que acertam os dez")
    X, y = ped.loc[DEZ, DUAS].to_numpy(float), ped.loc[DEZ, "cancelado"].to_numpy()
    w, b = pesos(m), m["b"]
    retas = {"do Caio: prazo = 9,5": (np.array([1.0, 0.0]), -9.5),
             "do Rafael: prazo − anteriores = 7,5": (np.array([1.0, -1.0]), -7.5),
             "de maior margem (SVM)": (w, b)}
    print("  reta                                  acerta   pedido mais perto   distância")
    for nome, (wr, br_) in retas.items():
        f = (X * wr).sum(axis=1) + br_
        dist = np.abs(f) / math.hypot(*wr)
        k = int(np.argmin(dist))
        print(f"  {nome:<36} {int(((f > 0) == (y == 1)).sum()):>2} de 10   {DEZ[k]:>17}   {br(dist[k]):>9}")
    novo = np.array([9.0, 2.0])
    lados = []
    for nome, (wr, br_) in retas.items():
        lados.append(f"{nome.split(':')[0]}: {'cancela' if (novo * wr).sum() + br_ > 0 else 'entregue'}")
    print(f"  um pedido novo com prazo 9 e 2 anteriores -> {'; '.join(lados)}")
    return retas


def parte3(ped, retas):
    print("\n3. As mesmas retas nos 414 pedidos de abril, com as duas colunas")
    abril = ped[ped["data_pedido"].str.startswith("2026-04")]
    X, y = abril[DUAS].to_numpy(float), sinal(abril["cancelado"])
    print(f"  {len(abril)} pedidos, {int((y > 0).sum())} cancelados")
    print("  regra                                 marca   cancelados pegos   entregues marcados   acertos")
    linhas = [("dizer «entregue» para todos", np.full(len(y), -1.0))]
    linhas += [(nome, (X * wr).sum(axis=1) + br_) for nome, (wr, br_) in retas.items()]
    for nome, f in linhas:
        pegos, falsos, erros = contar(f, y)
        print(f"  {nome:<36} {pegos + falsos:>6}   {pegos:>16}   {falsos:>18}   {len(y) - erros:>7}")


def parte4(ped):
    print("\n4. Margem flexível: os dez e mais dois pedidos que nenhuma reta acerta (1207 e 784)")
    ids = DEZ + TEIMOSOS
    X, y = ped.loc[ids, DUAS].to_numpy(float), sinal(ped.loc[ids, "cancelado"])
    for pid in TEIMOSOS:
        print(f"  pedido {pid}: prazo {int(ped.loc[pid, DUAS[0]])}, anteriores {int(ped.loc[pid, DUAS[1]])}, "
              f"cancelou {int(ped.loc[pid, 'cancelado'])}")
    gemeos = [int(p) for p in DEZ if (ped.loc[p, DUAS].to_numpy() == ped.loc[TEIMOSOS[0], DUAS].to_numpy()).all()]
    print(f"  o {TEIMOSOS[0]} tem o mesmo prazo e os mesmos anteriores do {', '.join(map(str, gemeos))}, que foi entregue")
    print("       C   w (prazo, anteriores)        b   largura   vetores de suporte   dentro da faixa   do lado errado")
    for C in (10.0, 1.0, 0.1, 0.01):
        m = treinar(X, y, k_linear, C)
        w, f = pesos(m), y * decisao(m, X)
        print(f"  {br(C):>6}   ({br(w[0], 3)}; {br(w[1], 3)})   {br(m['b'], 3):>7}   {br(2 / math.hypot(*w)):>7}"
              f"   {int((m['alfa'] > 0).sum()):>18}   {int(((f >= 0) & (f < 1 - 1e-6)).sum()):>15}   {int((f < 0).sum()):>14}")


def parte5(marco, abril):
    print("\n5. O mapa de entregas: três kernels, C = 1")
    X, y = marco[MAPA].to_numpy(float), sinal(marco["cancelado"])
    Xa, ya = abril[MAPA].to_numpy(float), sinal(abril["cancelado"])
    g = 1 / (X.shape[1] * X.var())
    print(f"  março: {len(y)} pedidos, {int((y > 0).sum())} cancelados; abril: {len(ya)} pedidos, "
          f"{int((ya > 0).sum())} cancelados")
    print(f"  gamma do RBF = 1 ÷ (2 colunas × variância {br(X.var())}) = {br(g, 5)}")
    kernels = {"linear": k_linear, "quadrático": k_quadratico, f"RBF, gamma {br(g, 3)}": lambda A, B: k_rbf(A, B, g)}
    print("  kernel              vetores de suporte   erros em março   erros em abril   onde corta a linha leste-oeste (km)")
    modelos = {}
    for nome, k in kernels.items():
        m = treinar(X, y, k, 1.0)
        modelos[nome] = m
        em, ea = contar(decisao(m, X), y)[2], contar(decisao(m, Xa), ya)[2]
        print(f"  {nome:<18} {int((m['alfa'] > 0).sum()):>20}   {em:>14}   {ea:>14}   "
              f"{texto_cortes(cortes(lambda P, m=m: decisao(m, P)))}")
    return modelos, g


def parte6(marco, modelos, g):
    print("\n6. O kernel por dentro: os pedidos 5013 e 5024")
    a, b = marco.loc[PAR[0], MAPA].to_numpy(float), marco.loc[PAR[1], MAPA].to_numpy(float)
    for pid, p in zip(PAR, (a, b)):
        print(f"  {pid}: leste {br(p[0], 1)}, norte {br(p[1], 1)}, cancelou {int(marco.loc[pid, 'cancelado'])}")
    pe = float((a * b).sum())
    fa, fb = phi([a])[0], phi([b])[0]
    print(f"  a · b = {br(pe)}; kernel quadrático (a · b)² = {br(pe ** 2)}")
    print(f"  colunas criadas: {PAR[0]} -> ({br(fa[0])}; {br(fa[1], 4)}; {br(fa[2])}), "
          f"{PAR[1]} -> ({br(fb[0])}; {br(fb[1], 4)}; {br(fb[2])})")
    print(f"  produto nas colunas criadas = {br(float((fa * fb).sum()))}")
    d2 = float(((a - b) ** 2).sum())
    print(f"  RBF: distância² = {br(d2)}; exp(−{br(g, 5)} × {br(d2)}) = {br(math.exp(-g * d2), 4)}")
    m = modelos["quadrático"]
    w = ((m["alfa"] * m["y"])[:, None] * phi(m["X"])).sum(axis=0)
    print(f"  o modelo quadrático escrito nas colunas criadas: f = {br(w[0], 4)} × leste² "
          f"{'−' if w[1] < 0 else '+'} {br(abs(w[1] * math.sqrt(2)), 4)} × leste × norte "
          f"{'+' if w[2] >= 0 else '−'} {br(abs(w[2]), 4)} × norte² {'−' if m['b'] < 0 else '+'} {br(abs(m['b']), 4)}")
    print(f"  na linha leste-oeste, f = 0 em leste = ±√({br(-m['b'], 4)} ÷ {br(w[0], 4)}) = "
          f"±{br(math.sqrt(-m['b'] / w[0]), 1)} km")
    X, y = marco[MAPA].to_numpy(float), sinal(marco["cancelado"])
    explicito = treinar(phi(X), y, k_linear, 1.0)
    dif = np.abs(decisao(explicito, phi(X)) - decisao(m, X)).max()
    print(f"  SVM linear nas três colunas criadas, contra o kernel quadrático: maior diferença de f "
          f"nos 80 pedidos {'abaixo de 0,0001' if dif < 1e-4 else '= ' + br(dif, 4)}")


def main():
    ped = carregar_pedidos()
    marco, abril = carregar_entregas()
    print(f"{len(ped)} pedidos em dados/loja/pedidos.csv; {len(marco) + len(abril)} entregas de motoboy em entregas.csv")
    m = parte1(ped)
    retas = parte2(ped, m)
    parte3(ped, retas)
    parte4(ped)
    modelos, g = parte5(marco, abril)
    parte6(marco, modelos, g)


if __name__ == "__main__":
    main()
