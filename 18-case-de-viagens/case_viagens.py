"""Case de viagens: os modelos do curso nas reservas da Agência Exemplo, com a mesma divisão e o mesmo preço.

Rode de dentro da pasta da aula:  python3 case_viagens.py

Também serve de módulo para o exercício:

    from case_viagens import ler, CANDIDATOS, nota_do_candidato, br, pct

A base é dados/viagens/reservas.csv, na raiz do kit. ler() traz só as reservas de 2025: as de 2026
são o teste, e só abrir_2026() lê essas, chamada pela parte 7.

A divisão segue o calendário. Cada trimestre de abril a dezembro de 2025 é medido por um modelo
treinado com as reservas cujo desfecho já existia no primeiro dia do trimestre: feitas antes e com a
viagem antes. A padronização e as categorias saem só desse treino.

Os modelos são os das aulas do curso, escritos aqui no NumPy: a logística da aula 09 (método de
Newton), o KNN da aula 13, o Naive Bayes da aula 14 e o SVM da aula 17 (método SMO). Não há sorteio:
tudo sai igual toda vez, menos os tempos da parte 6.
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parent / "dados" / "viagens" / "reservas.csv"
NUMERICAS = ["antecedencia_dias", "noites", "adultos", "criancas", "valor_total", "reservas_anteriores",
             "tarifa_flexivel"]
CATEGORIAS = ["destino", "canal", "pagamento"]
NASCE_DEPOIS = "valor_reembolsado"      # o financeiro só devolve depois que a reserva cancela
INICIO_TESTE = "2026-01-01"
TRIMESTRES = [("abr-jun", "2025-04-01", "2025-07-01"), ("jul-set", "2025-07-01", "2025-10-01"),
              ("out-dez", "2025-10-01", "2026-01-01")]

# O preço, igual para todos os modelos.
PRECO_LIGACAO = 20.0          # reais por ligação, acerte ou não
COMISSAO = 300.0              # o que a agência guarda quando a reserva vira remarcação
REMARCA = 1 / 3               # das reservas que iam cancelar e recebem ligação, 1 em 3 remarca
VALOR_ALCANCADO = COMISSAO * REMARCA       # R$ 100 por cancelamento que a ligação alcança
LIGACOES_POR_RESERVA = 1 / 5  # a equipe da Marina liga para no máximo 1 em cada 5 reservas

# A regra, escrita antes de rodar.
VANTAGEM_MINIMA = 1000.0      # o modelo só troca a lista de hoje se ganhar R$ 1.000 ou mais na soma
EMPATE = 500.0                # entre os modelos, até R$ 500 abaixo do melhor é empate
FAIXAS = [(1, 15, "até 15 dias"), (16, 60, "16 a 60 dias"), (61, 10_000, "mais de 60 dias")]


# ---------- leitura e divisão ----------

def _ler_tudo(caminho):
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com dados/ na raiz do kit.")
    base = pd.read_csv(caminho, parse_dates=["data_reserva", "data_viagem"])
    faltam = [c for c in ["id_reserva", "data_reserva", "data_viagem", "cancelou"] + NUMERICAS + CATEGORIAS
              if c not in base.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    return base.sort_values(["data_reserva", "id_reserva"]).reset_index(drop=True)


def ler(caminho=BASE, com_reembolso=False):
    """As reservas de 2025. As de 2026 saem aqui. A coluna que nasce depois só fica se pedir."""
    base = _ler_tudo(caminho)
    base = base[base["data_reserva"] < pd.Timestamp(INICIO_TESTE)].reset_index(drop=True)
    return base if com_reembolso else base.drop(columns=[NASCE_DEPOIS], errors="ignore")


def abrir_2026(caminho=BASE):
    base = _ler_tudo(caminho).drop(columns=[NASCE_DEPOIS], errors="ignore")
    return base[base["data_reserva"] >= pd.Timestamp(INICIO_TESTE)].reset_index(drop=True)


def treino_ate(base, inicio):
    """As reservas cujo desfecho já existia em `inicio`: feitas antes e com a viagem antes."""
    inicio = pd.Timestamp(inicio)
    return base[(base["data_reserva"] < inicio) & (base["data_viagem"] < inicio)]


def dobras(base):
    """[(nome, treino, deixadas de fora, medidas)] dos três trimestres de abril a dezembro de 2025."""
    saida = []
    for nome, ini, fim in TRIMESTRES:
        antes = base[base["data_reserva"] < pd.Timestamp(ini)]
        treino = treino_ate(base, ini)
        fora = antes.drop(index=treino.index)
        medidas = base[(base["data_reserva"] >= pd.Timestamp(ini)) & (base["data_reserva"] < pd.Timestamp(fim))]
        saida.append((nome, treino, fora, medidas))
    return saida


# ---------- preparação ----------

def colunas_numericas(p, curva=False, extra=()):
    x = p[NUMERICAS].astype(float).copy()
    if curva:
        log = np.log(x["antecedencia_dias"])
        x["antecedencia_dias"] = log
        x.insert(1, "log_antecedencia_ao_quadrado", log ** 2)
    for c in extra:
        x[c] = p[c].astype(float)
    return x


def preparar(treino, outros, curva=False, extra=()):
    """Matrizes do NumPy: as numéricas padronizadas pelo treino e as categorias em 0 e 1.

    Com curva=True, a antecedência entra como log e log ao quadrado. Média, desvio e categorias saem
    só do treino. Retorna (x do treino, [x de cada outro], nomes das colunas).
    """
    num = colunas_numericas(treino, curva, extra)
    media, desvio = num.mean(), num.std(ddof=0)
    categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}
    nomes = list(num.columns) + [f"{c}_{v}" for c in CATEGORIAS for v in categorias[c]]

    def matriz(p):
        partes = [((colunas_numericas(p, curva, extra) - media) / desvio).to_numpy(dtype=float)]
        for c in CATEGORIAS:
            for v in categorias[c]:
                partes.append((p[c] == v).to_numpy(dtype=float)[:, None])
        return np.hstack(partes)

    return matriz(treino), [matriz(o) for o in outros], nomes


def _produto(A, B):
    K = np.zeros((len(A), len(B)))
    for k in range(A.shape[1]):
        K += np.outer(A[:, k], B[:, k])
    return K


def _distancia2(A, B):
    K = np.zeros((len(A), len(B)))
    for k in range(A.shape[1]):
        K += (A[:, k][:, None] - B[:, k][None, :]) ** 2
    return K


# ---------- os modelos ----------

def treinar_logistica(X, y, penalidade=1.0, passos=30):
    """A logística da aula 09, pelo método de Newton, com a penalidade pequena. Intercepto primeiro."""
    m = np.c_[np.ones(len(X)), np.asarray(X, float)]
    y = np.asarray(y, float)
    d = m.shape[1]
    coef = np.zeros(d)
    pen = np.full(d, penalidade)
    pen[0] = 0
    for _ in range(passos):
        p = 1 / (1 + np.exp(-(m * coef[None, :]).sum(axis=1)))
        grad = (m * (p - y)[:, None]).sum(axis=0) + pen * coef
        w = p * (1 - p)
        hess = np.column_stack([(m * (w * m[:, k])[:, None]).sum(axis=0) for k in range(d)]) + np.diag(pen)
        coef = coef - np.linalg.solve(hess, grad)
    return coef


def nota_logistica(coef, X):
    m = np.c_[np.ones(len(X)), np.asarray(X, float)]
    return 1 / (1 + np.exp(-(m * np.asarray(coef)[None, :]).sum(axis=1)))


def nota_knn(X, y, Xn, k):
    """A parte de canceladas entre as k reservas do treino mais perto de cada reserva nova."""
    d = _distancia2(Xn, X)
    perto = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.asarray(y, float)[perto].mean(axis=1)


NB_GAUSS = ["antecedencia_dias", "noites", "valor_total", "reservas_anteriores"]   # em log(x + 1)
NB_CONTAGEM = ["destino", "canal", "pagamento", "tarifa_flexivel", "adultos", "criancas"]


def nota_bayes(treino, novas):
    """Naive Bayes da aula 14: gaussiana no log das numéricas e contagem com +1 nas outras.

    Retorna o log da razão: log P(cancela | reserva) − log P(não cancela | reserva).
    """
    y = treino["cancelou"].to_numpy()
    lados = []
    for classe in (1, 0):
        t = treino[y == classe]
        lp = np.full(len(novas), math.log(len(t) / len(treino)))
        for c in NB_GAUSS:
            v = np.log1p(t[c].astype(float))
            mu, dp = v.mean(), v.std(ddof=0)
            vn = np.log1p(novas[c].astype(float).to_numpy())
            lp += -0.5 * ((vn - mu) / dp) ** 2 - math.log(dp)
        for c in NB_CONTAGEM:
            valores = sorted(treino[c].unique())
            cont = t[c].value_counts()
            prob = {v: (cont.get(v, 0) + 1) / (len(t) + len(valores)) for v in valores}
            lp += np.log(novas[c].map(prob).fillna(1 / (len(t) + len(valores))).to_numpy(dtype=float))
        lados.append(lp)
    return lados[0] - lados[1]


def treinar_svm(X, y, kernel, C, tol=1e-3, max_passos=200_000):
    """O SVM da aula 17, pelo SMO. y em −1 e +1. Retorna X, y, alfa, b, kernel e passos."""
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
        raise RuntimeError(f"o SVM não parou em {max_passos} passos")
    v = y - u
    livres = (a > 0) & (a < C)
    if livres.any():
        b = float(v[livres].mean())
    else:
        de_baixo = [v[t] for t in range(n) if (a[t] == 0 and y[t] > 0) or (a[t] == C and y[t] < 0)]
        de_cima = [v[t] for t in range(n) if (a[t] == 0 and y[t] < 0) or (a[t] == C and y[t] > 0)]
        b = (max(de_baixo) + min(de_cima)) / 2
    return {"X": X, "y": y, "alfa": a, "b": b, "kernel": kernel, "passos": passo}


def k_rbf(gamma):
    return lambda A, B: np.exp(-gamma * _distancia2(A, B))


def gamma_scale(X):
    """A regra 'scale' da aula 17: 1 ÷ (colunas × variância de todos os números de X)."""
    return 1.0 / (X.shape[1] * np.asarray(X, float).var())


def decisao_svm(modelo, Xn):
    K = modelo["kernel"](modelo["X"], np.asarray(Xn, float))
    return ((modelo["alfa"] * modelo["y"])[:, None] * K).sum(axis=0) + modelo["b"]


# Os candidatos, na ordem do mais simples para o mais caro. É a ordem que desempata a escolha.
CANDIDATOS = [
    "lista de hoje: maior antecedência",
    "logística, antecedência em dias",
    "logística, antecedência em curva",
    "Naive Bayes",
    "KNN, K = 25",
    "KNN, K = 75",
    "KNN, K = 151",
    "SVM RBF, gamma × 0,25, C = 1",
]
LISTA_DE_HOJE = CANDIDATOS[0]


def nota_do_candidato(nome, treino, novas, info=None):
    """A nota de cada reserva nova pelo candidato `nome`, treinado em `treino`. Maior nota: liga antes.

    Se `info` for um dicionário, recebe o que o modelo guarda e quantas contas faz por reserva nova.
    """
    info = {} if info is None else info
    if nome == LISTA_DE_HOJE:
        info.update(guarda=0, contas=0)
        return novas["antecedencia_dias"].to_numpy(dtype=float)
    curva = nome == "logística, antecedência em curva"
    x, (xn,), cols = preparar(treino, [novas], curva=curva)
    y = treino["cancelou"].to_numpy()
    if nome.startswith("logística"):
        info.update(guarda=len(cols) + 1, contas=len(cols))
        return nota_logistica(treinar_logistica(x, y), xn)
    if nome == "Naive Bayes":
        n_par = 2 * 2 * len(NB_GAUSS) + 2 * sum(treino[c].nunique() for c in NB_CONTAGEM) + 2
        info.update(guarda=n_par, contas=2 * (len(NB_GAUSS) + len(NB_CONTAGEM)))
        return nota_bayes(treino, novas)
    if nome.startswith("KNN"):
        k = int(nome.split("= ")[1])
        info.update(guarda=len(treino) * len(cols), contas=len(treino) * len(cols))
        return nota_knn(x, y, xn, k)
    if nome.startswith("SVM"):
        mo = treinar_svm(x, np.where(y == 1, 1.0, -1.0), k_rbf(0.25 * gamma_scale(x)), 1.0)
        sv = int((mo["alfa"] > 0).sum())
        info.update(guarda=sv * len(cols), contas=sv * len(cols), vetores=sv)
        return decisao_svm(mo, xn)
    raise ValueError(f"candidato desconhecido: {nome}")


# ---------- medidas ----------

def lista(nota, ids, quantos):
    """As `quantos` reservas de nota mais alta; em empate, vem antes a de número menor."""
    ordem = np.lexsort((np.asarray(ids), -np.asarray(nota, float)))
    return ordem[:quantos]


def teto(n):
    return int(n * LIGACOES_POR_RESERVA)


def economia(pegas, ligacoes, preco=PRECO_LIGACAO, valor=VALOR_ALCANCADO):
    return valor * pegas - preco * ligacoes


def acerto_do_par(f, y):
    """Sorteando uma cancelada e uma não cancelada, em quantos pares a cancelada tem a nota maior."""
    posto = pd.Series(np.asarray(f, dtype=float)).rank().to_numpy()
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((posto[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def medir(nota, medidas):
    y = medidas["cancelou"].to_numpy()
    q = teto(len(medidas))
    pegas = int(y[lista(nota, medidas["id_reserva"].to_numpy(), q)].sum())
    return {"par": acerto_do_par(nota, y), "pegas": pegas, "ligacoes": q, "economia": economia(pegas, q)}


# ---------- texto ----------

def br(x, casas=2):
    if round(x, casas) == 0:
        x = 0.0
    s = f"{x:,.{casas}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, casas=1):
    return br(100 * x, casas) + "%"


def reais(x):
    return ("-R$ " if x < 0 else "R$ ") + br(abs(x), 0)


# ---------- a aula ----------

def cabecalho(base):
    c = int(base["cancelou"].sum())
    print(f"2025: {len(base)} reservas, {c} canceladas ({pct(c / len(base))}); 2026 fica fechado até a parte 7")
    print("colunas da hora da reserva: " + ", ".join(NUMERICAS + CATEGORIAS) + f"; fora: {NASCE_DEPOIS}")
    print("canceladas por antecedência: " + ", ".join(
        f"{rotulo} {pct(base.loc[base['antecedencia_dias'].between(a, b), 'cancelou'].mean())}" for a, b, rotulo in FAIXAS))
    print()


def parte1(ds):
    print("1. A divisão: cada trimestre medido pelo que já tinha desfecho no primeiro dia dele")
    print(f"  {'trimestre':<11}{'treino':>8}{'canceladas':>12}{'fora':>7}{'canceladas':>12}"
          f"{'medidas':>9}{'canceladas':>12}{'teto':>6}")
    for nome, tr, fora, me in ds:
        print(f"  {nome:<11}{len(tr):>8}{pct(tr['cancelou'].mean()):>12}{len(fora):>7}{pct(fora['cancelou'].mean()):>12}"
              f"{len(me):>9}{int(me['cancelou'].sum()):>12}{teto(len(me)):>6}")
    print("  treino: feitas antes do trimestre e com a viagem antes dele; fora: feitas antes, com a viagem depois")
    print("  teto: 1 ligação para cada 5 reservas do trimestre medido")
    print()


def parte2(ds):
    print("2. O preço, igual para todos os modelos")
    print(f"  ligação: {reais(PRECO_LIGACAO)}; reserva que ia cancelar e recebe ligação: 1 em 3 remarca, "
          f"e a remarcação guarda {reais(COMISSAO)}")
    print(f"  economia = {reais(VALOR_ALCANCADO)} × canceladas na lista − {reais(PRECO_LIGACAO)} × ligações")
    print(f"  ligar só vale a pena para quem tem mais de {pct(PRECO_LIGACAO / VALOR_ALCANCADO, 0)} de chance de cancelar")
    print(f"  {'trimestre':<11}{'ligações':>10}{'lista a esmo':>14}{'lista perfeita':>16}")
    for nome, _, _, me in ds:
        q = teto(len(me))
        esmo = economia(q * me["cancelou"].mean(), q)
        perfeita = economia(min(q, int(me["cancelou"].sum())), q)
        print(f"  {nome:<11}{q:>10}{reais(esmo):>14}{reais(perfeita):>16}")
    print("  lista a esmo: a média de ligar para reservas sorteadas; lista perfeita: só para quem cancelou")
    print()


def parte3(ds):
    print(f"3. A coluna que nasce depois: {NASCE_DEPOIS}")
    base = ler(com_reembolso=True)
    pos = base[NASCE_DEPOIS] > 0
    print(f"  reembolso maior que zero: {int((pos & (base['cancelou'] == 1)).sum())} canceladas e "
          f"{int((pos & (base['cancelou'] == 0)).sum())} não canceladas")
    por_id = base.set_index("id_reserva")
    linhas = []
    for nome, tr, _, me in ds:
        tr_r, me_r = por_id.loc[tr["id_reserva"]].reset_index(), por_id.loc[me["id_reserva"]].reset_index()
        x, (xm,), _ = preparar(tr_r, [me_r], extra=[NASCE_DEPOIS])
        nota = nota_logistica(treinar_logistica(x, tr_r["cancelou"]), xm)
        linhas.append(medir(nota, me_r))
    print("  logística, antecedência em dias, com o reembolso: acerto do par "
          + ", ".join(pct(l["par"]) for l in linhas) + f"; economia somada {reais(sum(l['economia'] for l in linhas))}")
    print("  essa linha não entra na comparação")
    print()


def rodar_candidatos(ds):
    return {nome: [medir(nota_do_candidato(nome, tr, me), me) for _, tr, _, me in ds] for nome in CANDIDATOS}


def parte4(ds, res):
    print("4. Os candidatos nas três dobras: economia de cada trimestre, com o teto de ligações")
    print(f"  {'modelo':<36}{'abr-jun':>10}{'jul-set':>10}{'out-dez':>10}{'soma':>10}{'na lista':>10}{'acerto do par':>15}")
    for nome, linhas in res.items():
        print(f"  {nome:<36}" + "".join(f"{reais(l['economia']):>10}" for l in linhas)
              + f"{reais(sum(l['economia'] for l in linhas)):>10}{sum(l['pegas'] for l in linhas):>10}"
              + f"{pct(np.mean([l['par'] for l in linhas])):>15}")
    total_lig = sum(teto(len(me)) for _, _, _, me in ds)
    total_canc = sum(int(me["cancelou"].sum()) for _, _, _, me in ds)
    print(f"  na lista: canceladas entre as {total_lig} ligações das três dobras ({total_canc} canceladas ao todo)")
    print("  acerto do par: a média das três dobras")
    print()


def escolher(res):
    soma = {n: sum(l["economia"] for l in res[n]) for n in CANDIDATOS}
    modelos = CANDIDATOS[1:]
    melhor = max(soma[n] for n in modelos)
    empatados = [n for n in modelos if melhor - soma[n] <= EMPATE]
    escolhido = empatados[0]                  # CANDIDATOS já está na ordem do mais simples
    troca = soma[escolhido] - soma[LISTA_DE_HOJE] >= VANTAGEM_MINIMA
    return escolhido, empatados, soma, troca


def parte5(res):
    escolhido, empatados, soma, troca = escolher(res)
    print("5. A escolha")
    print(f"  regra, escrita antes de rodar: entre os modelos, a maior economia somada; até {reais(EMPATE)} abaixo "
          "é empate, e o empate fica com o mais simples (a ordem da tabela)")
    print(f"  o modelo só troca a lista de hoje se ganhar dela por {reais(VANTAGEM_MINIMA)} ou mais na soma")
    print("  empatados com o melhor: " + "; ".join(empatados))
    print(f"  escolhido: {escolhido}, {reais(soma[escolhido])}")
    print(f"  lista de hoje: {reais(soma[LISTA_DE_HOJE])}; diferença {reais(soma[escolhido] - soma[LISTA_DE_HOJE])}")
    print(f"  fica {'o modelo: ' + escolhido if troca else 'a lista de hoje'}")
    print()
    return escolhido if troca else LISTA_DE_HOJE


def _mediana(fn, rodadas=3):
    ts = []
    for _ in range(rodadas):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def parte6(base):
    tr = treino_ate(base, INICIO_TESTE)
    print(f"6. O custo: o que cada modelo guarda e quantas contas faz por reserva nova, com o treino de 2026 "
          f"({len(tr)} reservas)")
    print(f"  {'modelo':<36}{'números do modelo':>19}{'contas por reserva':>20}{'segundos para treinar':>23}")
    teste = base.iloc[:200]
    for nome in CANDIDATOS:
        info = {}
        nota_do_candidato(nome, tr, teste, info)
        seg = _mediana(lambda: nota_do_candidato(nome, tr, teste))
        print(f"  {nome:<36}{br(info['guarda'], 0):>19}{br(info['contas'], 0):>20}{br(seg, 3):>23}")
    print("  segundos: treinar e dar nota a 200 reservas, mediana de 3 rodadas; mudam de um computador para outro")
    print()


def parte7(base, escolhido):
    novas = abrir_2026()
    tr = treino_ate(base, INICIO_TESTE)
    q = teto(len(novas))
    y = novas["cancelou"].to_numpy()
    print(f"7. 2026, uma vez: {len(novas)} reservas de janeiro a março, {int(y.sum())} canceladas "
          f"({pct(y.mean())}); teto de {q} ligações")
    print(f"  treino: as {len(tr)} reservas com desfecho em 1º de janeiro ({pct(tr['cancelou'].mean())} canceladas)")
    print(f"  {'modelo':<36}{'economia':>10}{'na lista':>10}{'acerto do par':>15}")
    notas = {}
    for nome in CANDIDATOS:
        notas[nome] = nota_do_candidato(nome, tr, novas)
        m = medir(notas[nome], novas)
        marca = "  <- escolhido" if nome == escolhido else ""
        print(f"  {nome:<36}{reais(m['economia']):>10}{m['pegas']:>10}{pct(m['par']):>15}{marca}")
    esmo = economia(q * y.mean(), q)
    print(f"  lista a esmo: {reais(esmo)}; lista perfeita: {reais(economia(min(q, int(y.sum())), q))}")
    flex_25 = base["tarifa_flexivel"].mean()
    flex_26 = novas["tarifa_flexivel"].mean()
    print(f"  tarifa flexível: {pct(flex_25)} das reservas de 2025 e {pct(flex_26)} das de 2026")
    if escolhido.startswith("logística"):
        print(f"  nota média do escolhido em 2026: {pct(notas[escolhido].mean())}; canceladas de verdade: {pct(y.mean())}")
    print()
    return novas, notas[escolhido], q


def parte8(novas, nota, q, escolhido):
    y = novas["cancelou"].to_numpy()
    na_lista = np.zeros(len(novas), dtype=bool)
    na_lista[lista(nota, novas["id_reserva"].to_numpy(), q)] = True
    print(f"8. Os erros do escolhido em 2026 ({escolhido})")
    print(f"  {'antecedência':<17}{'tarifa':<11}{'reservas':>9}{'canceladas':>12}{'ligações':>10}"
          f"{'pegas':>7}{'perdidas':>10}{'ligações à toa':>16}")
    for a, b, rotulo in FAIXAS:
        for flex in (0, 1):
            g = (novas["antecedencia_dias"].between(a, b) & (novas["tarifa_flexivel"] == flex)).to_numpy()
            pegas = int((y[g] == 1)[na_lista[g]].sum())
            canc = int(y[g].sum())
            lig = int(na_lista[g].sum())
            print(f"  {rotulo:<17}{'flexível' if flex else 'comum':<11}{int(g.sum()):>9}{canc:>12}{lig:>10}"
                  f"{pegas:>7}{canc - pegas:>10}{lig - pegas:>16}")
    pegas = int(y[na_lista].sum())
    print(f"  {'total':<28}{len(novas):>9}{int(y.sum()):>12}{q:>10}{pegas:>7}{int(y.sum()) - pegas:>10}{q - pegas:>16}")
    print("  perdidas: canceladas fora da lista; ligações à toa: na lista e não cancelaram")


def main():
    base = ler()
    ds = dobras(base)
    cabecalho(base)
    parte1(ds)
    parte2(ds)
    parte3(ds)
    res = rodar_candidatos(ds)
    parte4(ds, res)
    escolhido = parte5(res)
    parte6(base)
    novas, nota, q = parte7(base, escolhido)
    parte8(novas, nota, q, escolhido)


if __name__ == "__main__":
    main()
