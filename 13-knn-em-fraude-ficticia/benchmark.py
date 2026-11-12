"""Escolher o K e o corte do voto do KNN nas dobras de janeiro a maio, medir o tempo e abrir junho uma vez.

Rode de dentro da pasta da aula:

    python3 benchmark.py                # escolhe nas dobras, mede o tempo e grava benchmark.json
    python3 benchmark.py --abrir-teste  # refaz a escolha, compara com o registro e mede junho, uma vez

Rodar `python3 benchmark.py` de novo, com o registro já gravado, refaz as contas e compara número por
número. Igual, ele diz igual. Diferente, ele lista o que mudou e sai com código 1. Os tempos da parte 5
dependem do computador: eles ficam no registro, mas fora da comparação.

O KNN é o da aula «KNN e dimensionalidade»: distância de Pitágoras nas cinco colunas, cada uma dividida
pelo desvio das compras que o modelo conhece. Uma compra vai para análise quando pelo menos `corte` dos
K vizinhos foram fraude. O custo de uma regra numa dobra é o preço da análise vezes as compras marcadas,
mais o valor de cada fraude que passou sem análise.

Junho só é lido dentro de abrir_teste(): nenhuma conta da escolha vê essas compras.
"""
import hashlib
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
ARQUIVO = AQUI.parent / "dados" / "fraude" / "transacoes.csv"
REGISTRO = AQUI / "benchmark.json"
COLUNAS = ["valor", "idade_conta_dias", "tentativas_cartao", "distancia_entrega_km", "hora"]
N_DOBRAS = 5
SORTEIO = 13
GRADE_K = [1, 3, 5, 7, 11, 15, 21, 31, 51]
PRECO_ANALISE = 15.0
REGRA = ("menor custo médio por dobra; empate exato fica com menos alarmes falsos e, depois, "
         "com o menor K")


# ---------- dados ----------

def _ler(caminho=ARQUIVO):
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com dados/ na raiz do kit.")
    dados = pd.read_csv(caminho)
    faltam = [c for c in COLUNAS + ["id_transacao", "data_hora", "fraude"] if c not in dados.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    if dados[COLUNAS + ["fraude"]].isna().any().any():
        raise ValueError("há valores vazios nas colunas usadas")
    return dados.sort_values(["data_hora", "id_transacao"]).reset_index(drop=True)


def digital(caminho=ARQUIVO):
    """O sha256 do arquivo: muda se uma letra do CSV mudar."""
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def carregar_treino(caminho=ARQUIVO):
    """As compras de janeiro a maio. Junho fica de fora aqui."""
    dados = _ler(caminho)
    return dados[~dados["data_hora"].str.startswith("2026-06-")].reset_index(drop=True)


def _carregar_teste(caminho=ARQUIVO):
    dados = _ler(caminho)
    return dados[dados["data_hora"].str.startswith("2026-06-")].reset_index(drop=True)


# ---------- as dobras ----------

def dobras_estratificadas(fraude, n_dobras, sorteio):
    """A dobra (de 0 a n_dobras − 1) de cada compra, com as fraudes repartidas por igual.

    Embaralha as posições das compras legítimas com np.random.default_rng(sorteio).permutation e, com o
    MESMO gerador, as posições das fraudes. Em cada grupo, a posição que sai em i-ésimo lugar fica na
    dobra i % n_dobras. Assim cada dobra recebe o mesmo número de fraudes, com diferença de no máximo 1.
    """
    fraude = np.asarray(fraude)
    if not set(np.unique(fraude).tolist()) <= {0, 1}:
        raise ValueError("o rótulo de fraude precisa ser 0 ou 1")
    if n_dobras < 2 or n_dobras > int(fraude.sum()):
        raise ValueError(f"o número de dobras vai de 2 até o número de fraudes ({int(fraude.sum())}); veio {n_dobras}")
    gerador = np.random.default_rng(sorteio)
    dobras = np.empty(len(fraude), dtype=int)
    for classe in (0, 1):
        posicoes = gerador.permutation(np.flatnonzero(fraude == classe))
        dobras[posicoes] = np.arange(len(posicoes)) % n_dobras
    return dobras


def dobras_sem_separar(n, n_dobras, sorteio):
    """Para comparar: o sorteio comum, sem olhar o rótulo."""
    dobras = np.empty(n, dtype=int)
    dobras[np.random.default_rng(sorteio).permutation(n)] = np.arange(n) % n_dobras
    return dobras


# ---------- o KNN ----------

def votos(conhecidas, rotulos, novas, ks):
    """Para cada compra nova, quantos dos K vizinhos foram fraude, para cada K de ks.

    Uma conta de distância e uma ordenação por compra servem para todos os K. No empate de
    distância, vale a ordem das compras conhecidas (ordenação estável).
    """
    conhecidas = np.asarray(conhecidas, dtype=float)
    rotulos = np.asarray(rotulos)
    saida = np.empty((len(novas), len(ks)), dtype=int)
    for i, linha in enumerate(np.asarray(novas, dtype=float)):
        ordem = np.argsort(((conhecidas - linha) ** 2).sum(axis=1), kind="stable")
        acumulado = np.cumsum(rotulos[ordem[:max(ks)]])
        saida[i] = acumulado[[k - 1 for k in ks]]
    return saida


def padronizar(conhecidas, novas):
    """Divide cada coluna pelo desvio das compras conhecidas. A média some na distância."""
    desvio = conhecidas.std(ddof=0)
    if (desvio == 0).any():
        raise ValueError(f"coluna sem variação: {', '.join(desvio.index[desvio == 0])}")
    return (conhecidas / desvio).to_numpy(float), (novas / desvio).to_numpy(float)


def regras(ks):
    """Os pares (K, corte) da grade: maioria, pelo menos 2 e pelo menos 1 vizinho fraude."""
    return [(k, c) for k in ks for c in sorted({k // 2 + 1, 2, 1}, reverse=True) if c <= k]


def custo_da_regra(votos_k, fraude, valor, corte, preco):
    """Marcadas, fraudes pegas, alarmes falsos, valor das fraudes que passaram e o custo em reais."""
    votos_k, fraude, valor = np.asarray(votos_k), np.asarray(fraude), np.asarray(valor, dtype=float)
    if not (len(votos_k) == len(fraude) == len(valor)):
        raise ValueError(f"tamanhos diferentes: {len(votos_k)} votos, {len(fraude)} rótulos, {len(valor)} valores")
    if corte < 1:
        raise ValueError("o corte é um número de vizinhos, de 1 para cima")
    if preco < 0:
        raise ValueError("o preço da análise não pode ser negativo")
    marcada = votos_k >= corte
    passou = float(valor[(~marcada) & (fraude == 1)].sum())
    return {"marcadas": int(marcada.sum()), "pegas": int((marcada & (fraude == 1)).sum()),
            "alarmes": int((marcada & (fraude == 0)).sum()), "passou": passou,
            "custo": preco * int(marcada.sum()) + passou}


# ---------- o benchmark ----------

def benchmark(preco=PRECO_ANALISE, sorteio=SORTEIO, dividir=dobras_estratificadas, custo=custo_da_regra,
              caminho=ARQUIVO):
    """Roda a escolha inteira só com janeiro a maio e devolve o registro, sem gravar."""
    treino = carregar_treino(caminho)
    fraude = treino["fraude"].to_numpy()
    valor = treino["valor"].to_numpy(float)
    dobras = np.asarray(dividir(fraude, N_DOBRAS, sorteio))
    if len(dobras) != len(treino) or sorted(set(dobras.tolist())) != list(range(N_DOBRAS)):
        raise ValueError(f"as dobras precisam ser 0 a {N_DOBRAS - 1}, uma para cada compra")
    v = np.empty((len(treino), len(GRADE_K)), dtype=int)
    for j in range(N_DOBRAS):
        fora = dobras != j
        conh, novas = padronizar(treino.loc[fora, COLUNAS], treino.loc[~fora, COLUNAS])
        v[~fora] = votos(conh, fraude[fora], novas, GRADE_K)
    resultados = {}
    for k, c in regras(GRADE_K):
        col = v[:, GRADE_K.index(k)]
        por_dobra = [custo(col[dobras == j], fraude[dobras == j], valor[dobras == j], c, preco)
                     for j in range(N_DOBRAS)]
        resultados[f"{k}:{c}"] = {campo: [d[campo] for d in por_dobra] for campo in
                                  ("marcadas", "pegas", "alarmes", "passou", "custo")}
    nunca = [float(valor[(dobras == j) & (fraude == 1)].sum()) for j in range(N_DOBRAS)]
    escolha = escolher(resultados)
    ids = treino["id_transacao"].to_numpy()
    texto_dobras = ";".join(",".join(str(i) for i in sorted(ids[dobras == j])) for j in range(N_DOBRAS))
    sem_separar = [int(fraude[dobras_sem_separar(len(treino), N_DOBRAS, s) == j].sum())
                   for s in range(20) for j in range(N_DOBRAS)]
    return {
        "pergunta": f"K e corte do voto do KNN de fraude, com análise a R$ {br(preco, 2)}",
        "dados": {"arquivo": "dados/fraude/transacoes.csv", "sha256": digital(caminho),
                  "treino": "2026-01-01 a 2026-05-31", "compras_treino": int(len(treino)),
                  "fraudes_treino": int(fraude.sum()), "teste": "junho de 2026, fechado até abrir_teste"},
        "colunas": COLUNAS,
        "preco_analise": preco,
        "dobras": {"n": N_DOBRAS, "sorteio": sorteio, "tamanhos": [int((dobras == j).sum()) for j in range(N_DOBRAS)],
                   "fraudes": [int(fraude[dobras == j].sum()) for j in range(N_DOBRAS)],
                   "sha256_dos_ids": hashlib.sha256(texto_dobras.encode()).hexdigest(),
                   "sem_separar_sorteios_0_a_19": [min(sem_separar), max(sem_separar)]},
        "grade_k": GRADE_K,
        "regras": [f"{k}:{c}" for k, c in regras(GRADE_K)],
        "regra_da_escolha": REGRA,
        "resultados": resultados,
        "nunca_analisar": nunca,
        "escolha": escolha,
        "versoes": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "registrado_em": date.today().isoformat(),
        "tempos": None,
        "teste": None,
    }


def escolher(resultados):
    """A regra: menor custo médio; empate exato, menos alarmes; depois, menor K."""
    def chave(nome):
        k, c = (int(x) for x in nome.split(":"))
        r = resultados[nome]
        return (round(float(np.mean(r["custo"])), 6), sum(r["alarmes"]), k, -c)
    nome = min(resultados, key=chave)
    k, c = (int(x) for x in nome.split(":"))
    return {"k": k, "corte": c, "custo_medio": float(np.mean(resultados[nome]["custo"]))}


# ---------- o tempo ----------

def medir_tempos(treino, k):
    """Milissegundos por compra nova, mediana de 3 rodadas. Depende do computador.

    A base guardada é repetida 10 e 100 vezes só para medir o tempo: nenhuma escolha usa essas cópias.
    """
    x, _ = padronizar(treino[COLUNAS], treino[COLUNAS])
    novas = x[np.random.default_rng(0).choice(len(x), 100, replace=False)]

    def ms(base, n, passo):
        rodadas = []
        for _ in range(3):
            t = time.perf_counter()
            for linha in novas[:n]:
                passo(((base - linha) ** 2).sum(axis=1))
            rodadas.append((time.perf_counter() - t) / n * 1000)
        return float(np.median(rodadas))

    def ordenar(kk):
        return lambda d: np.argsort(d, kind="stable")[:kk]

    def separar(kk):
        return lambda d: np.argpartition(d, kk - 1)[:kk]

    por_k = {str(kk): {"ordenar": ms(x, 100, ordenar(kk)), "separar": ms(x, 100, separar(kk))}
             for kk in (1, k, 51)}
    por_tamanho = {}
    for vezes, n in ((1, 100), (10, 100), (100, 20)):
        base = np.tile(x, (vezes, 1))
        por_tamanho[str(len(base))] = {"distancias": ms(base, n, lambda d: None),
                                       "ordenar": ms(base, n, ordenar(k)), "separar": ms(base, n, separar(k)),
                                       "megabytes": base.nbytes / 1e6}
    return {"k": k, "por_k": por_k, "por_tamanho": por_tamanho}


# ---------- comparar com o registro ----------

FORA_DA_COMPARACAO = ("versoes", "registrado_em", "tempos", "teste")


def diferencas(registro, agora, caminho=""):
    """Lista o que mudou entre o registro e a rodada de agora; números iguais até 6 casas valem como iguais."""
    if isinstance(registro, dict) and isinstance(agora, dict):
        saida = []
        for chave in sorted(set(registro) | set(agora)):
            if not caminho and chave in FORA_DA_COMPARACAO:
                continue
            nome = f"{caminho}.{chave}" if caminho else chave
            if chave not in registro or chave not in agora:
                saida.append(f"{nome}: só existe {'agora' if chave in agora else 'no registro'}")
            else:
                saida += diferencas(registro[chave], agora[chave], nome)
        return saida
    if isinstance(registro, list) and isinstance(agora, list) and len(registro) == len(agora):
        return [d for i, (a, b) in enumerate(zip(registro, agora)) for d in diferencas(a, b, f"{caminho}[{i}]")]
    if isinstance(registro, (int, float)) and isinstance(agora, (int, float)) \
            and not isinstance(registro, bool) and not isinstance(agora, bool):
        return [] if round(float(registro), 6) == round(float(agora), 6) else [f"{caminho}: registro {registro}, agora {agora}"]
    return [] if registro == agora else [f"{caminho}: registro {registro}, agora {agora}"]


def avisar_versoes(registro):
    agora = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for nome, versao in registro.get("versoes", {}).items():
        if agora.get(nome) != versao:
            print(f"aviso: o registro foi feito com {nome} {versao}; aqui é {agora.get(nome)}")


# ---------- números com vírgula ----------

def br(x, casas=0):
    if round(x, casas) == 0:
        x = 0.0
    s = f"{x:,.{casas}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def ms_br(x):
    """Milissegundos com dois algarismos que importam."""
    casas = 0 if x >= 10 else 1 if x >= 1 else 2 if x >= 0.1 else 3
    return br(x, casas)


def nome_corte(k, c):
    return "maioria" if c == k // 2 + 1 and c > 2 else f"pelo menos {c}"


# ---------- o que o programa mostra ----------

def mostrar(reg):
    d, dob, preco = reg["dados"], reg["dobras"], reg["preco_analise"]
    res = reg["resultados"]
    print(f"treino: {d['compras_treino']} compras de janeiro a maio de 2026 ({d['fraudes_treino']} fraudes); "
          "junho fica fechado")
    print(f"dados: {d['arquivo']}, sha256 {d['sha256'][:12]}")
    print(f"colunas: {', '.join(reg['colunas'])}, divididas pelo desvio das compras que o modelo conhece")
    print(f"preços: análise R$ {br(preco, 2)} por compra marcada; fraude que passa custa o valor da compra")
    print()
    print(f"1. As dobras: {dob['n']}, com as fraudes repartidas por igual, pelo sorteio {dob['sorteio']}")
    print("  dobra   " + "".join(f"{j + 1:>8}" for j in range(dob["n"])))
    print("  compras " + "".join(f"{t:>8}" for t in dob["tamanhos"]))
    print("  fraudes " + "".join(f"{t:>8}" for t in dob["fraudes"]))
    a, b = dob["sem_separar_sorteios_0_a_19"]
    print(f"  sem separar por classe, nos sorteios 0 a 19, uma dobra ficava com {a} a {b} fraudes")
    print(f"  o código das dobras é {dob['sha256_dos_ids'][:12]}")
    print()
    print("2. A maioria dos vizinhos decide, como no predict do scikit-learn")
    print(f"  {'K':>3}  {'fraudes pegas':>13}  {'alarmes':>7}  {'fraude que passou':>17}  {'custo médio por dobra':>21}")
    for k in reg["grade_k"]:
        r = res[f"{k}:{k // 2 + 1}"]
        print(f"  {k:>3}  {sum(r['pegas']):>7} de {d['fraudes_treino']:<3}  {sum(r['alarmes']):>7}  "
              f"{'R$ ' + br(sum(r['passou'])):>17}  {'R$ ' + br(np.mean(r['custo'])):>21}")
    print(f"  nunca analisar: R$ {br(np.mean(reg['nunca_analisar']))} por dobra")
    print()
    print("3. O corte do voto: custo médio por dobra")
    print(f"  {'K':>3}  {'maioria':>9}  {'pelo menos 2':>12}  {'pelo menos 1':>12}  "
          f"{'pegas e alarmes com pelo menos 1':>32}")
    for k in reg["grade_k"]:
        def celula(c, largura):
            return f"{'R$ ' + br(np.mean(res[f'{k}:{c}']['custo'])):>{largura}}" if f"{k}:{c}" in res else f"{'-':>{largura}}"
        um = res[f"{k}:1"]
        print(f"  {k:>3}  {celula(k // 2 + 1, 9)}  {celula(2, 12)}  {celula(1, 12)}  "
              f"{sum(um['pegas']):>17} e {sum(um['alarmes']):>3} alarmes")
    print()
    e = reg["escolha"]
    escolhido = res[f"{e['k']}:{e['corte']}"]
    print("4. A escolha")
    print(f"  regra, escrita antes de rodar: {reg['regra_da_escolha']}")
    print(f"  escolhido: K = {e['k']}, {nome_corte(e['k'], e['corte'])} {'vizinho' if e['corte'] == 1 else 'vizinhos'} "
          f"fraude, custo médio R$ {br(e['custo_medio'])} por dobra")
    print(f"  por dobra: {', '.join('R$ ' + br(x) for x in escolhido['custo'])}")
    print(f"  pegou {sum(escolhido['pegas'])} de {d['fraudes_treino']} fraudes, com {sum(escolhido['alarmes'])} "
          f"alarmes falsos: {br(sum(escolhido['marcadas']) / dob['n'], 1)} análises por dobra")
    print("  as cinco regras mais baratas, custo em cada dobra:")
    ordem = sorted(res, key=lambda n: (round(float(np.mean(res[n]["custo"])), 6), sum(res[n]["alarmes"]),
                                       int(n.split(":")[0])))
    for nome in ordem[:5]:
        k2, c2 = (int(x) for x in nome.split(":"))
        r = res[nome]
        print(f"    K = {k2:<3}{nome_corte(k2, c2):<13} média R$ {br(np.mean(r['custo'])):>4} | "
              + "  ".join(f"{br(x):>5}" for x in r["custo"]))
    print()


def mostrar_tempos(t):
    k = t["k"]
    print("5. O tempo de uma previsão, no seu computador (milissegundos por compra, mediana de 3 rodadas)")
    print(f"  com as 6647 compras guardadas   {'ordenar todas':>13}  {'separar só as K':>15}")
    for kk, v in t["por_k"].items():
        print(f"  K = {kk:<27} {ms_br(v['ordenar']):>13}  {ms_br(v['separar']):>15}")
    print(f"  com K = {k}")
    print(f"  {'compras guardadas':>17}  {'memória':>9}  {'só as distâncias':>16}  {'ordenar todas':>13}  {'separar só as K':>15}")
    for n, v in t["por_tamanho"].items():
        print(f"  {br(int(n)):>17}  {br(v['megabytes'], 1) + ' MB':>9}  {ms_br(v['distancias']):>16}  "
              f"{ms_br(v['ordenar']):>13}  {ms_br(v['separar']):>15}")
    print("  66.470 e 664.700 são as mesmas compras repetidas 10 e 100 vezes, só para medir o tempo")
    print()


def rodar_benchmark(registro, preco, dividir, custo):
    inicio = time.perf_counter()
    reg = benchmark(preco, SORTEIO, dividir, custo)
    segundos = time.perf_counter() - inicio
    mostrar(reg)
    treino = carregar_treino()
    reg["tempos"] = medir_tempos(treino, reg["escolha"]["k"])
    reg["tempos"]["benchmark_segundos"] = segundos
    mostrar_tempos(reg["tempos"])
    n = len(reg["regras"])
    print("6. O registro")
    print(f"  as {n} regras saíram de uma conta de distância por compra, em {br(segundos, 1)} segundos")
    if not registro.exists():
        registro.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  gravei {registro.name}: dados, dobras, grade, regra, os custos das {n} regras, a escolha, "
              "os tempos e as versões")
        print("  junho continua fechado")
        return 0
    antigo = json.loads(registro.read_text(encoding="utf-8"))
    avisar_versoes(antigo)
    mudou = diferencas(antigo, reg)
    if mudou:
        print(f"  diferente de {registro.name} em {len(mudou)} {'ponto' if len(mudou) == 1 else 'pontos'}:")
        for linha in mudou[:20]:
            print(f"    {linha}")
        return 1
    print(f"  igual a {registro.name}: os mesmos dados, as mesmas dobras e os custos das {n} regras")
    print("  junho já foi aberto para esse registro" if antigo.get("teste") else "  junho continua fechado")
    return 0


def abrir_teste(registro, preco, dividir, custo):
    """Refaz a escolha, exige que ela bata com o registro e só então mede junho, uma vez."""
    if not registro.exists():
        print(f"não achei {registro.name}. Rode a escolha antes (sem --abrir-teste): o teste só abre depois "
              "da escolha gravada")
        return 1
    antigo = json.loads(registro.read_text(encoding="utf-8"))
    avisar_versoes(antigo)
    reg = benchmark(preco, antigo["dobras"]["sorteio"], dividir, custo)
    mudou = diferencas(antigo, reg)
    if mudou:
        print(f"refiz a escolha, e ela não bate com {registro.name} em {len(mudou)} "
              f"{'ponto' if len(mudou) == 1 else 'pontos'}:")
        for linha in mudou[:20]:
            print(f"  {linha}")
        print("o teste não abre com um registro que não bate com a conta")
        return 1
    if antigo.get("teste"):
        t = antigo["teste"]
        print(f"junho já foi aberto para esse registro em {t['aberto_em']}: K = {t['k']}, "
              f"{t['pegas']} de {t['fraudes']} fraudes, custo R$ {br(t['custo'])}")
        print("o teste abre uma vez. Para outra pergunta, comece outro registro")
        return 1
    print(f"refiz a escolha e comparei com {registro.name}: igual")
    k, corte = antigo["escolha"]["k"], antigo["escolha"]["corte"]
    treino, teste = carregar_treino(), _carregar_teste()
    conh, novas = padronizar(treino[COLUNAS], teste[COLUNAS])
    inicio = time.perf_counter()
    v = votos(conh, treino["fraude"].to_numpy(), novas, [k])[:, 0]
    segundos = time.perf_counter() - inicio
    fraude, valor = teste["fraude"].to_numpy(), teste["valor"].to_numpy(float)
    r = custo(v, fraude, valor, corte, preco)
    nunca = float(valor[fraude == 1].sum())
    print(f"modelo final: K = {k}, {nome_corte(k, corte)} {'vizinho' if corte == 1 else 'vizinhos'} fraude, "
          f"com as {len(treino)} compras de janeiro a maio guardadas")
    print(f"junho, aberto agora: {len(teste)} compras, {int(fraude.sum())} fraudes")
    print(f"  marcou {r['marcadas']}: {r['pegas']} fraudes e {r['alarmes']} alarmes falsos")
    print(f"  fraude que passou: R$ {br(r['passou'], 2)}")
    print(f"  custo R$ {br(r['custo'], 2)}; nas dobras, a mesma regra deu R$ {br(antigo['escolha']['custo_medio'])} "
          "por dobra")
    print(f"  nunca analisar custaria R$ {br(nunca, 2)}")
    print(f"  tempo: {br(segundos * 1000 / len(teste), 2)} ms por compra, no seu computador")
    antigo["teste"] = {"aberto_em": date.today().isoformat(), "k": k, "corte": corte, "compras": int(len(teste)),
                       "fraudes": int(fraude.sum()), "marcadas": r["marcadas"], "pegas": r["pegas"],
                       "alarmes": r["alarmes"], "passou": r["passou"], "custo": r["custo"],
                       "nunca_analisar": nunca, "ms_por_compra": segundos * 1000 / len(teste)}
    registro.write_text(json.dumps(antigo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"gravei o teste em {registro.name}; junho não abre de novo para esse registro")
    return 0


def rodar(argv, registro=REGISTRO, preco=PRECO_ANALISE, dividir=dobras_estratificadas, custo=custo_da_regra):
    if argv not in ([], ["--abrir-teste"]):
        print(f"use: python3 {Path(sys.argv[0]).name} [--abrir-teste]")
        return 2
    if argv == ["--abrir-teste"]:
        return abrir_teste(Path(registro), preco, dividir, custo)
    return rodar_benchmark(Path(registro), preco, dividir, custo)


if __name__ == "__main__":
    sys.exit(rodar(sys.argv[1:]))
