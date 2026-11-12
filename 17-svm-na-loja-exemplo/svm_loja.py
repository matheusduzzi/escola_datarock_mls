"""SVM no cancelamento da Loja Exemplo: kernels, escala e custo, só com o pandas e o NumPy.

Rode de dentro da pasta da aula:  python3 svm_loja.py

Também serve de módulo para o exercício:

    from svm_loja import ler, dobras, preparar, k_linear, k_rbf, gamma_scale, treinar, decisao

A base é dados/loja/pedidos_export_completo.csv, na raiz do kit. Só entram as colunas que existem
na hora do pedido, como na aula 07: dias_ate_entrega e contatos_sac ficam fora. ler() já tira junho,
que é o teste. Só abrir_junho() lê junho, e só a parte 7 chama abrir_junho().

As dobras seguem o calendário: cada mês de fevereiro a maio é medido pelo modelo treinado nos meses
anteriores. A padronização e as categorias saem só do treino de cada dobra.

O SVM é treinado aqui mesmo, pela função treinar(), a mesma da aula 16: ela mexe em dois alfas por
vez (o método SMO). A regressão logística é a da aula 09, pelo método de Newton. Não há sorteio:
tudo sai igual toda vez, menos os tempos da parte 6.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parent / "dados" / "loja" / "pedidos_export_completo.csv"
NUMERICAS = ["valor", "prazo_prometido_dias", "pedidos_anteriores", "cliente_novo"]
CATEGORIAS = ["categoria", "forma_pagamento", "canal"]
NASCEM_DEPOIS = ["dias_ate_entrega", "contatos_sac"]      # gravadas depois do pedido
INICIO_TESTE = "2026-06-01"
MESES = {1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio", 6: "junho"}
LIGACOES_POR_PEDIDO = 1 / 4        # a equipe do Caio: no máximo 1 ligação para cada 4 pedidos
MAX_PASSOS = 20_000                # o treino desiste depois disso
TOL_FINA = 1e-6                    # para ver quanto a parada mexe no resultado
PAR = (1978, 1545)                # dois pedidos de abril para a conta da distância
VANTAGEM_MINIMA = 0.01             # o SVM só troca a logística se ganhar 1 ponto ou mais
EMPATE = 0.005                     # entre SVM, diferença até meio ponto é empate
TAMANHOS = [300, 600, 1200, 1950]  # para medir o custo

# A grade, escrita antes de rodar. gamma × 1 é a regra 'scale' do scikit-learn.
GRADE = ([("reta", None, c) for c in (0.01, 0.1, 1.0)]
         + [("polinomial", None, c) for c in (0.01, 0.1, 1.0)]
         + [("RBF", m, c) for m in (0.25, 1.0, 4.0) for c in (0.1, 1.0, 10.0)])
ORDEM_DE_CUSTO = {"reta": 0, "polinomial": 1, "RBF": 2}


# ---------- leitura e preparação ----------

def _ler_tudo(caminho):
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com dados/ na raiz do kit.")
    base = pd.read_csv(caminho, parse_dates=["data_pedido"])
    faltam = [c for c in ["id_pedido", "data_pedido", "cancelado"] + NUMERICAS + CATEGORIAS if c not in base.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    return base.drop(columns=[c for c in NASCEM_DEPOIS if c in base.columns])


def ler(caminho=BASE):
    """Janeiro a maio, em ordem de data e de pedido. Junho sai aqui."""
    base = _ler_tudo(caminho)
    base = base[base["data_pedido"] < pd.Timestamp(INICIO_TESTE)]
    return base.sort_values(["data_pedido", "id_pedido"]).reset_index(drop=True)


def abrir_junho(caminho=BASE):
    base = _ler_tudo(caminho)
    base = base[base["data_pedido"] >= pd.Timestamp(INICIO_TESTE)]
    return base.sort_values(["data_pedido", "id_pedido"]).reset_index(drop=True)


def dobras(base):
    """[(nome do mês medido, treino, medida)]: fevereiro a maio, cada um com os meses anteriores."""
    mes = base["data_pedido"].dt.month
    return [(MESES[m], base[mes < m], base[mes == m]) for m in (2, 3, 4, 5)]


def preparar(treino, outros, escala=True):
    """Matrizes do NumPy: as quatro numéricas (padronizadas pelo treino, se escala) e as categorias em 0 e 1.

    Retorna (x_treino, [x de cada outro], colunas). Média, desvio e categorias saem só do treino.
    """
    media = treino[NUMERICAS].mean()
    desvio = treino[NUMERICAS].std(ddof=0)
    categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}
    colunas = NUMERICAS + [f"{c}_{v}" for c in CATEGORIAS for v in categorias[c]]

    def matriz(p):
        x = (p[NUMERICAS] - media) / desvio if escala else p[NUMERICAS].astype(float)
        partes = [x.to_numpy(dtype=float)]
        for c in CATEGORIAS:
            for v in categorias[c]:
                partes.append((p[c] == v).to_numpy(dtype=float)[:, None])
        return np.hstack(partes)

    return matriz(treino), [matriz(o) for o in outros], colunas


def sinal(rotulo):
    """0 e 1 viram −1 e +1, como na aula 16."""
    return np.where(np.asarray(rotulo) == 1, 1.0, -1.0)


# ---------- kernels ----------
# Coluna por coluna: são só 14 colunas, e a conta fica igual em qualquer computador.

def _produto(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    K = np.zeros((len(A), len(B)))
    for k in range(A.shape[1]):
        K += np.outer(A[:, k], B[:, k])
    return K


def _distancia2(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    K = np.zeros((len(A), len(B)))
    for k in range(A.shape[1]):
        K += (A[:, k][:, None] - B[:, k][None, :]) ** 2
    return K


def k_linear(A, B):
    return _produto(A, B)


def k_polinomial(gamma, coef0=1.0):
    """(gamma × a · b + 1)²: o quadrático da aula 16 com o termo constante (pergunta 2 de lá)."""
    return lambda A, B: (gamma * _produto(A, B) + coef0) ** 2


def k_rbf(gamma):
    """exp(−gamma × distância²), como na aula 16."""
    return lambda A, B: np.exp(-gamma * _distancia2(A, B))


def gamma_scale(X):
    """A regra 'scale' do scikit-learn: 1 ÷ (número de colunas × variância de todos os números de X)."""
    X = np.asarray(X, float)
    return 1.0 / (X.shape[1] * X.var())


def kernel_da_grade(nome, multiplicador, X):
    g = gamma_scale(X)
    if nome == "reta":
        return k_linear, None
    if nome == "polinomial":
        return k_polinomial(g), g
    return k_rbf(g * multiplicador), g * multiplicador


# ---------- treino e decisão ----------

class NaoParou(RuntimeError):
    pass


def treinar(X, y, kernel, C, tol=1e-3, max_passos=MAX_PASSOS):
    """SVM pelo problema dual, pelo SMO da aula 16. y em −1 e +1.

    Para quando a maior violação das condições de ótimo fica abaixo de tol (o mesmo 0,001 do
    scikit-learn). Se passar de max_passos, levanta NaoParou. Retorna X, y, alfa, b, kernel, C e passos.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    K = kernel(X, X)
    diag = np.diag(K).copy()
    a = np.zeros(n)
    u = np.zeros(n)
    zero = 1e-12 * C
    for passo in range(1, max_passos + 1):
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
        raise NaoParou(f"o treino não parou em {max_passos} passos")
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
    """f(x) = soma de alfa × y × K(pedido do treino, x) + b. Positivo: cancela."""
    K = modelo["kernel"](modelo["X"], np.asarray(X, float))
    return ((modelo["alfa"] * modelo["y"])[:, None] * K).sum(axis=0) + modelo["b"]


def vetores_de_suporte(modelo):
    return int((modelo["alfa"] > 0).sum())


def pesos(modelo):
    """w = soma de alfa × y × x, só para a reta: f(x) = w · x + b."""
    return ((modelo["alfa"] * modelo["y"])[:, None] * modelo["X"]).sum(axis=0)


# ---------- a logística da aula 09 ----------

def _score(m, coef):
    return (m * np.asarray(coef)[None, :]).sum(axis=1)


def treinar_logistica(X, y, penalidade=1.0, passos=30):
    """Regressão logística pelo método de Newton, com a penalidade pequena da aula 09. Intercepto primeiro."""
    m = np.c_[np.ones(len(X)), np.asarray(X, float)]
    y = np.asarray(y, float)
    d = m.shape[1]
    coef = np.zeros(d)
    pen = np.full(d, penalidade)
    pen[0] = 0
    for _ in range(passos):
        p = 1 / (1 + np.exp(-_score(m, coef)))
        grad = (m * (p - y)[:, None]).sum(axis=0) + pen * coef
        w = p * (1 - p)
        hess = np.column_stack([(m * (w * m[:, k])[:, None]).sum(axis=0) for k in range(d)]) + np.diag(pen)
        coef = coef - np.linalg.solve(hess, grad)
    return coef


def nota_logistica(coef, X):
    m = np.c_[np.ones(len(X)), np.asarray(X, float)]
    return 1 / (1 + np.exp(-_score(m, coef)))


# ---------- medidas ----------

def acerto_do_par(f, y):
    """Sorteando um cancelado e um entregue, em quantos pares o cancelado tem o f maior (a área sob a ROC)."""
    posto = pd.Series(np.asarray(f, dtype=float)).rank().to_numpy()
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((posto[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def teto(n):
    return int(n * LIGACOES_POR_PEDIDO)


def _pegos(f, y, ids, quantos):
    ordem = np.lexsort((np.asarray(ids), -np.asarray(f, float)))   # f maior primeiro; empate pelo pedido
    return int(np.asarray(y)[ordem[:quantos]].sum())


# ---------- texto ----------

def br(x, casas=2):
    if round(x, casas) == 0:
        x = 0.0
    s = f"{x:,.{casas}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, casas=1):
    return br(100 * x, casas) + "%"


def nome_modelo(nome, mult, C):
    if nome == "reta":
        return f"reta, C = {br(C, 2)}"
    if nome == "polinomial":
        return f"polinomial, C = {br(C, 2)}"
    return f"RBF, gamma × {br(mult, 2)}, C = {br(C, 2)}"


# ---------- a aula ----------

def cabecalho(base):
    cancelados = int(base["cancelado"].sum())
    print(f"janeiro a maio: {len(base)} pedidos, {cancelados} cancelados ({pct(cancelados / len(base))}); "
          f"junho fica fechado até a parte 7")
    print("colunas da hora do pedido: " + ", ".join(NUMERICAS + CATEGORIAS) + "; fora: " + ", ".join(NASCEM_DEPOIS))
    print()


def parte1(ds):
    print("1. As dobras: cada mês medido pelo modelo treinado nos meses anteriores")
    print(f"  {'mês medido':<11}{'treino':<22}{'pedidos':>8}{'cancelados':>12}{'medidos':>9}{'cancelados':>12}{'teto':>6}")
    for nome, tr, me in ds:
        meses = sorted(tr["data_pedido"].dt.month.unique())
        rotulo = MESES[meses[0]] if len(meses) == 1 else f"{MESES[meses[0]]} a {MESES[meses[-1]]}"
        print(f"  {nome:<11}{rotulo:<22}{len(tr):>8}{int(tr['cancelado'].sum()):>12}"
              f"{len(me):>9}{int(me['cancelado'].sum()):>12}{teto(len(me)):>6}")
    print("  teto: 1 ligação para cada 4 pedidos do mês medido")
    print()


def parte2(ds):
    nome, tr, me = ds[2]                       # abril
    print(f"2. A escala: a dobra de {nome}, treino de janeiro a março ({len(tr)} pedidos)")
    media = tr[NUMERICAS].mean()
    desvio = tr[NUMERICAS].std(ddof=0)
    print(f"  {'coluna':<22}{'média':>9}{'desvio':>9}")
    for c in NUMERICAS:
        print(f"  {c:<22}{br(media[c]):>9}{br(desvio[c]):>9}")
    a = me[me["id_pedido"] == PAR[0]].iloc[0]
    b = me[me["id_pedido"] == PAR[1]].iloc[0]
    for p in (a, b):
        print(f"  pedido {p['id_pedido']}: valor {br(p['valor'])}, prazo {p['prazo_prometido_dias']}, "
              f"anteriores {p['pedidos_anteriores']}, cliente novo {p['cliente_novo']}, "
              f"{p['categoria']}, {p['forma_pagamento']}, {p['canal']}, cancelou {p['cancelado']}")
    par = me.set_index("id_pedido").loc[list(PAR)].reset_index()
    _, (par_sem,), _ = preparar(tr, [par], escala=False)
    _, (par_com,), _ = preparar(tr, [par], escala=True)
    for rotulo, par in (("sem padronizar", par_sem), ("padronizado", par_com)):
        d = (par[0] - par[1]) ** 2
        partes = [f"{c} {br(d[k])}" for k, c in enumerate(NUMERICAS)]
        partes.append(f"categorias {br(d[len(NUMERICAS):].sum())}")
        print(f"  distância² {rotulo}: {br(d.sum())} = " + " + ".join(partes)
              + f"; o valor responde por {pct(d[0] / d.sum())}")
    y = sinal(tr["cancelado"])
    for escala in (False, True):
        x, (xm,), _ = preparar(tr, [me], escala=escala)
        rotulo = "padronizada" if escala else "sem padronizar"
        try:
            mo = treinar(x, y, k_linear, 0.01)
            print(f"  reta do SVM (C = 0,01), {rotulo}: parou em {mo['passos']} passos; "
                  f"acerto do par em {nome} {pct(acerto_do_par(decisao(mo, xm), me['cancelado']))}")
        except NaoParou:
            print(f"  reta do SVM (C = 0,01), {rotulo}: não parou em {br(MAX_PASSOS, 0)} passos")
    for escala in (False, True):
        x, (xm,), _ = preparar(tr, [me], escala=escala)
        g = gamma_scale(x)
        mo = treinar(x, y, k_rbf(g), 1.0)
        rotulo = "padronizado" if escala else "sem padronizar"
        print(f"  RBF (gamma 'scale' = {br(g, 6) if g < 0.001 else br(g, 4)}, C = 1), {rotulo}: "
              f"{mo['passos']} passos, {vetores_de_suporte(mo)} vetores de suporte, "
              f"acerto do par em {nome} {pct(acerto_do_par(decisao(mo, xm), me['cancelado']))}")
    print()


def rodar_grade(ds):
    """Os 15 SVM e a logística em cada dobra. Retorna {modelo: [(acerto, pegos, passos, vetores)]}."""
    res = {"logística": []}
    for cfg in GRADE:
        res[cfg] = []
    for _, tr, me in ds:
        x, (xm,), _ = preparar(tr, [me])
        y = sinal(tr["cancelado"])
        ym, ids, q = me["cancelado"].to_numpy(), me["id_pedido"].to_numpy(), teto(len(me))
        nt = nota_logistica(treinar_logistica(x, tr["cancelado"]), xm)
        res["logística"].append((acerto_do_par(nt, ym), _pegos(nt, ym, ids, q), None, None, int((nt > 0.5).sum())))
        for cfg in GRADE:
            k, _g = kernel_da_grade(cfg[0], cfg[1], x)
            mo = treinar(x, y, k, cfg[2])
            f = decisao(mo, xm)
            res[cfg].append((acerto_do_par(f, ym), _pegos(f, ym, ids, q), mo["passos"], vetores_de_suporte(mo),
                             int((f > 0).sum())))
    return res


def parte3(ds, res):
    total_teto = sum(teto(len(me)) for _, _, me in ds)
    total_canc = sum(int(me["cancelado"].sum()) for _, _, me in ds)
    print("3. Os 15 SVM e a logística nas quatro dobras: acerto do par em cada mês medido")
    print(f"  {'modelo':<29}{'fev':>7}{'mar':>7}{'abr':>7}{'mai':>7}{'média':>8}"
          f"{'no teto':>9}{'f > 0':>7}{'passos':>8}")
    for cfg, linhas in res.items():
        nome = "logística (aula 09)" if cfg == "logística" else nome_modelo(*cfg)
        acertos = [l[0] for l in linhas]
        pegos = sum(l[1] for l in linhas)
        passos = "" if cfg == "logística" else str(sum(l[2] for l in linhas))
        print(f"  {nome:<29}" + "".join(f"{pct(a):>7}" for a in acertos)
              + f"{pct(np.mean(acertos)):>8}{pegos:>9}{sum(l[4] for l in linhas):>7}{passos:>8}")
    print(f"  no teto: cancelados entre os {total_teto} pedidos de f mais alto, somando as quatro dobras "
          f"({total_canc} cancelados ao todo)")
    print("  f > 0: pedidos do lado «cancela» pelo sinal do f, somando as dobras; na logística, nota acima de 0,5")
    print("  gamma × 1 é a regra 'scale' do scikit-learn, calculada no treino de cada dobra")
    print()


def escolher(res):
    svm = [(cfg, float(np.mean([l[0] for l in res[cfg]]))) for cfg in GRADE]
    melhor = max(m for _, m in svm)
    empatados = [cfg for cfg, m in svm if melhor - m <= EMPATE]
    escolhido = min(empatados, key=lambda c: (ORDEM_DE_CUSTO[c[0]], c[2], c[1] or 0))
    return escolhido, dict(svm), empatados


def parte4(res, ds):
    escolhido, medias, empatados = escolher(res)
    m_log = float(np.mean([l[0] for l in res["logística"]]))
    print("4. A escolha")
    print("  regra, escrita antes de rodar: entre os 15 SVM, o maior acerto do par médio; até meio ponto abaixo "
          "é empate, e o empate fica com o mais barato (reta, depois polinomial, depois RBF; o menor C)")
    print("  o SVM só troca a logística se ganhar dela por 1 ponto ou mais")
    print(f"  empatados com o melhor: " + "; ".join(nome_modelo(*c) for c in empatados))
    print(f"  escolhido: {nome_modelo(*escolhido)}, média {pct(medias[escolhido])}")
    print(f"  logística: média {pct(m_log)}; diferença {br(100 * (medias[escolhido] - m_log), 1)} ponto")
    fica = "o SVM" if medias[escolhido] - m_log >= VANTAGEM_MINIMA else "a logística"
    print(f"  fica {fica}")
    finos = []
    for _, tr, me in ds:
        x, (xm,), _ = preparar(tr, [me])
        k, _g = kernel_da_grade(escolhido[0], escolhido[1], x)
        mo = treinar(x, sinal(tr["cancelado"]), k, escolhido[2], tol=TOL_FINA, max_passos=10 * MAX_PASSOS)
        finos.append(acerto_do_par(decisao(mo, xm), me["cancelado"]))
    print(f"  o escolhido, treinado até a violação ficar abaixo de {br(TOL_FINA, 6)} (e não de 0,001): "
          + ", ".join(pct(a) for a in finos) + f"; média {pct(np.mean(finos))}")
    print()
    return escolhido


def melhor_rbf(res):
    rbf = [cfg for cfg in GRADE if cfg[0] == "RBF"]
    return max(rbf, key=lambda c: (np.mean([l[0] for l in res[c]]), -c[2]))


def _mb(n):
    return n * n * 8 / 1e6


def parte5(base, escolhido, rbf):
    print("5. O custo contado: sai igual em qualquer computador")
    print(f"  {'pedidos':>8}  {'modelo':<29}{'passos':>8}{'vetores de suporte':>20}"
          f"{'tabela do kernel':>18}{'contas por pedido novo':>24}")
    linhas = []
    for n in TAMANHOS:
        tr = base.iloc[:n]
        x, _, colunas = preparar(tr, [])
        y = sinal(tr["cancelado"])
        for cfg in (escolhido, rbf):
            k, _g = kernel_da_grade(cfg[0], cfg[1], x)
            mo = treinar(x, y, k, cfg[2])
            sv = vetores_de_suporte(mo)
            contas = len(colunas) if cfg[0] == "reta" else sv * len(colunas)
            linhas.append((n, cfg, mo["passos"], sv))
            print(f"  {n:>8}  {nome_modelo(*cfg):<29}{mo['passos']:>8}{sv:>20}"
                  f"{br(_mb(n), 1) + ' MB':>18}{br(contas, 0):>24}")
    print(f"  a logística guarda 15 números e faz 14 contas por pedido novo, com qualquer número de pedidos")
    print(f"  com 100.000 pedidos, a tabela do kernel inteira teria {br(_mb(100_000) / 1000, 0)} GB")
    print()
    return linhas


def _mediana_tempo(fn, rodadas=3):
    ts = []
    for _ in range(rodadas):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def parte6(base, escolhido, rbf):
    print("6. O custo medido: segundos de treino no seu computador (mediana de 3 rodadas)")
    print(f"  {'pedidos':>8}{'logística':>12}{'reta do SVM':>14}{'RBF':>10}")
    for n in TAMANHOS:
        tr = base.iloc[:n]
        y = sinal(tr["cancelado"])

        def com_preparo(f):
            return lambda: f(preparar(tr, [])[0])

        t_log = _mediana_tempo(com_preparo(lambda x: treinar_logistica(x, tr["cancelado"])))
        t_ret = _mediana_tempo(com_preparo(lambda x: treinar(x, y, kernel_da_grade(*escolhido[:2], x)[0], escolhido[2])))
        t_rbf = _mediana_tempo(com_preparo(lambda x: treinar(x, y, kernel_da_grade(*rbf[:2], x)[0], rbf[2])))
        print(f"  {n:>8}{br(t_log, 3):>12}{br(t_ret, 3):>14}{br(t_rbf, 3):>10}")
    print("  os números mudam de um computador para outro e de uma rodada para outra")
    print()


def parte7(base, escolhido):
    junho = abrir_junho()
    x, (xj,), _ = preparar(base, [junho])
    yj, ids, q = junho["cancelado"].to_numpy(), junho["id_pedido"].to_numpy(), teto(len(junho))
    mo = treinar(x, sinal(base["cancelado"]), kernel_da_grade(*escolhido[:2], x)[0], escolhido[2])
    f = decisao(mo, xj)
    nt = nota_logistica(treinar_logistica(x, base["cancelado"]), xj)
    print(f"7. Junho, uma vez: {len(junho)} pedidos, {int(yj.sum())} cancelados; teto de {q} ligações")
    print(f"  treino: os {len(base)} pedidos de janeiro a maio, padronizados pela média e pelo desvio deles")
    print(f"  {'modelo':<29}{'acerto do par':>15}{'cancelados no teto':>20}")
    print(f"  {'logística (aula 09)':<29}{pct(acerto_do_par(nt, yj)):>15}{_pegos(nt, yj, ids, q):>20}")
    print(f"  {nome_modelo(*escolhido):<29}{pct(acerto_do_par(f, yj)):>15}{_pegos(f, yj, ids, q):>20}")
    topo_l = set(ids[np.lexsort((ids, -nt))[:q]])
    topo_s = set(ids[np.lexsort((ids, -f))[:q]])
    print(f"  pedidos nas duas listas de ligação: {len(topo_l & topo_s)} de {q}")
    print(f"  o f do SVM em junho vai de {br(f.min(), 4)} a {br(f.max(), 4)}; "
          f"{int((f > 0).sum())} pedidos com f positivo")
    print(f"  {'f de junho, em quartos':<24}{'pedidos':>8}{'f de':>9}{'f até':>9}{'cancelaram':>12}"
          f"{'nota média da logística':>25}")
    ordem = np.lexsort((ids, f))
    for k, parte in enumerate(np.array_split(ordem, 4), start=1):
        print(f"  {str(k) + 'º quarto':<24}{len(parte):>8}{br(f[parte].min(), 4):>9}{br(f[parte].max(), 4):>9}"
              f"{pct(yj[parte].mean()):>12}{pct(nt[parte].mean()):>25}")
    print(f"  w do modelo final: maior peso {br(np.abs(pesos(mo)).max(), 4)}; b = {br(mo['b'], 4)}")
    print(f"  vetores de suporte do modelo final: {vetores_de_suporte(mo)} de {len(base)} pedidos")


def main():
    base = ler()
    ds = dobras(base)
    cabecalho(base)
    parte1(ds)
    parte2(ds)
    res = rodar_grade(ds)
    parte3(ds, res)
    escolhido = parte4(res, ds)
    rbf = melhor_rbf(res)
    parte5(base, escolhido, rbf)
    parte6(base, escolhido, rbf)
    parte7(base, escolhido)


if __name__ == "__main__":
    main()
