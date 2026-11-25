"""Naive Bayes nas mensagens do atendimento e nas compras com cartão da Loja Exemplo.

Só o pandas e o NumPy. Rode de dentro da pasta da aula:  python3 bayes.py

Também serve de módulo para o exercício:

    from bayes import (br, carregar_mensagens, separar_mensagens, palavras, contar,
                       multinomial_log, probabilidades, carregar_compras, separar_compras,
                       ajustar_gaussiana, gaussiana_prob)

As mensagens e as compras de janeiro a maio são as que o modelo conhece, com o rótulo. As de
junho fazem o papel de casos novos. Não há sorteio: a saída sai igual toda vez.
"""
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
MENSAGENS = AQUI.parent / "dados" / "mensagens" / "mensagens.csv"
COMPRAS = AQUI.parent / "dados" / "fraude" / "transacoes.csv"
ASSUNTOS = ["atraso", "troca", "pagamento", "outro"]
COLUNAS = ["valor", "idade_conta_dias", "tentativas_cartao", "distancia_entrega_km", "hora"]
CURTO = {"valor": "valor", "idade_conta_dias": "conta", "tentativas_cartao": "tent",
         "distancia_entrega_km": "dist", "hora": "hora"}
EM_LOG = ["valor", "idade_conta_dias", "distancia_entrega_km"]

# Mensagens novas que o Diego mandou. Elas não estão no CSV.
MAO = "pix não chegou"
MISTURA = "a entrega está atrasada e quero devolver o produto"
REPETIDA = "não chegou"
MAO_VOCAB = ["chegou", "nao", "pedido", "pix"]
MAO_ASSUNTOS = ["atraso", "pagamento"]


def br(x, casas=2):
    """Número com vírgula decimal e ponto de milhar, como na aula."""
    if round(x, casas) == 0:
        x = 0.0
    s = f"{x:,.{casas}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, casas=1):
    return br(100 * x, casas) + "%"


# ---------------------------------------------------------------- mensagens

def palavras(texto):
    """Minúsculas, sem acento, só letras: «Não CHEGOU!» vira ['nao', 'chegou']."""
    if not isinstance(texto, str):
        raise TypeError("palavras recebe um texto")
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z]+", t)


def carregar_mensagens(caminho=MENSAGENS):
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com dados/ ao lado.")
    dados = pd.read_csv(caminho)
    faltam = [c for c in ["id_mensagem", "data", "texto", "assunto"] if c not in dados.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    return dados


def separar_mensagens(dados):
    """Janeiro a maio (o que o modelo conhece) e junho (as mensagens novas)."""
    junho = dados["data"].str.startswith("2026-06-")
    return dados[~junho], dados[junho]


def contar(listas, assuntos):
    """As contagens que o Naive Bayes guarda. Ele não aprende mais nada.

    listas: uma lista de palavras por mensagem. assuntos: o assunto de cada uma.
    Retorna um dicionário com:
      n_msgs[a]       quantas mensagens têm o assunto a;
      n_palavras[a]   Counter: quantas vezes cada palavra aparece nas mensagens de a;
      n_com[a]        Counter: em quantas mensagens de a cada palavra aparece (uma vez por mensagem);
      vocab           as palavras vistas, em ordem alfabética.
    """
    listas, assuntos = list(listas), list(assuntos)
    if len(listas) != len(assuntos):
        raise ValueError("uma lista de palavras para cada assunto")
    rotulos = [a for a in ASSUNTOS if a in set(assuntos)] + sorted(set(assuntos) - set(ASSUNTOS))
    modelo = {"assuntos": rotulos, "n_msgs": Counter(), "n_palavras": {a: Counter() for a in rotulos},
              "n_com": {a: Counter() for a in rotulos}}
    for ps, a in zip(listas, assuntos):
        modelo["n_msgs"][a] += 1
        modelo["n_palavras"][a].update(ps)
        modelo["n_com"][a].update(set(ps))
    modelo["vocab"] = sorted({p for a in rotulos for p in modelo["n_palavras"][a]})
    return modelo


def multinomial_log(ps, modelo, alpha=1.0):
    """log P(assunto) + soma, palavra por palavra (repetida conta de novo), de log P(palavra | assunto).

    P(palavra | assunto) = (vezes da palavra no assunto + alpha) ÷ (palavras do assunto + alpha × vocabulário).
    Palavra fora do vocabulário fica de fora. Com alpha = 0, uma palavra nunca vista no assunto zera o
    assunto: o log vira -inf.
    """
    if isinstance(ps, str):
        raise TypeError("passe a lista de palavras, não o texto: use palavras(texto)")
    if alpha < 0:
        raise ValueError("alpha não pode ser negativo")
    total = sum(modelo["n_msgs"].values())
    v = len(modelo["vocab"])
    vocab = set(modelo["vocab"])
    logs = {}
    for a in modelo["assuntos"]:
        s = math.log(modelo["n_msgs"][a] / total)
        tam = sum(modelo["n_palavras"][a].values())
        for p in ps:
            if p not in vocab:
                continue
            q = (modelo["n_palavras"][a][p] + alpha) / (tam + alpha * v)
            s += math.log(q) if q > 0 else -math.inf
        logs[a] = s
    return logs


def bernoulli_log(ps, modelo, alpha=1.0):
    """log P(assunto) + soma, para cada palavra do vocabulário, de log P(aparece) ou log P(não aparece).

    P(aparece | assunto) = (mensagens do assunto com a palavra + alpha) ÷ (mensagens do assunto + 2 × alpha).
    A palavra repetida conta uma vez. A palavra que falta entra com log(1 − P(aparece | assunto)).
    """
    if isinstance(ps, str):
        raise TypeError("passe a lista de palavras, não o texto: use palavras(texto)")
    if alpha <= 0:
        raise ValueError("alpha precisa ser maior que zero")
    total = sum(modelo["n_msgs"].values())
    tem = set(ps)
    logs = {}
    for a in modelo["assuntos"]:
        n = modelo["n_msgs"][a]
        s = math.log(n / total)
        for p in modelo["vocab"]:
            q = (modelo["n_com"][a][p] + alpha) / (n + 2 * alpha)
            s += math.log(q) if p in tem else math.log(1 - q)
        logs[a] = s
    return logs


def reduzir(modelo, vocab, assuntos):
    """O mesmo modelo, só com algumas palavras e alguns assuntos: para fazer a conta à mão."""
    r = {"assuntos": list(assuntos), "vocab": sorted(vocab),
         "n_msgs": Counter({a: modelo["n_msgs"][a] for a in assuntos}),
         "n_palavras": {a: Counter({p: modelo["n_palavras"][a][p] for p in vocab}) for a in assuntos},
         "n_com": {a: Counter({p: modelo["n_com"][a][p] for p in vocab}) for a in assuntos}}
    return r


def probabilidades(logs):
    """Dos logs para probabilidades que somam 1 (dividindo cada exp pelo total)."""
    finitos = [x for x in logs.values() if x != -math.inf]
    if not finitos:
        raise ValueError("todos os assuntos ficaram com probabilidade zero")
    m = max(finitos)
    e = {a: (math.exp(x - m) if x != -math.inf else 0.0) for a, x in logs.items()}
    z = sum(e.values())
    return {a: e[a] / z for a in logs}


def _melhor(probs):
    return max(probs, key=probs.get)


def _linha_probs(probs):
    return "  ".join(f"{a} {pct(probs[a])}" for a in probs)


# ---------------------------------------------------------------- compras

def carregar_compras(caminho=COMPRAS):
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com dados/ ao lado.")
    dados = pd.read_csv(caminho)
    faltam = [c for c in COLUNAS + ["id_transacao", "data_hora", "fraude"] if c not in dados.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    return dados.set_index("id_transacao", drop=False)


def separar_compras(dados):
    junho = dados["data_hora"].str.startswith("2026-06-")
    return dados[~junho], dados[junho]


def em_log(dados):
    """valor, idade da conta e distância da entrega em log (log de 1 + x), o resto igual."""
    d = dados[COLUNAS].astype(float).copy()
    for c in EM_LOG:
        d[c] = np.log1p(d[c])
    return d


def ajustar_gaussiana(x, y):
    """Para cada classe e cada coluna: média e variância. Mais a proporção de cada classe.

    A variância ganha 1e-9 vezes a maior variância das colunas, como o var_smoothing do scikit-learn.
    """
    x = pd.DataFrame(x).astype(float)
    y = pd.Series(np.asarray(y), index=x.index)
    extra = 1e-9 * float(x.var(ddof=0).max())
    m = {"colunas": list(x.columns), "classes": sorted(y.unique().tolist())}
    for k in m["classes"]:
        parte = x[y == k]
        m[k] = {"prior": float((y == k).mean()), "media": parte.mean(), "var": parte.var(ddof=0) + extra}
    return m


def gaussiana_log(x, m):
    """Um log por classe e por linha: log P(classe) + soma, coluna por coluna, do log da curva normal."""
    x = pd.DataFrame(x)[m["colunas"]].astype(float)
    out = {}
    for k in m["classes"]:
        mu, var = m[k]["media"], m[k]["var"]
        termos = -0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)
        out[k] = math.log(m[k]["prior"]) + termos.sum(axis=1)
    return pd.DataFrame(out)


def gaussiana_prob(x, m, classe=1):
    """P(classe | colunas) para cada linha."""
    lg = gaussiana_log(x, m)
    mx = lg.max(axis=1)
    e = np.exp(lg.sub(mx, axis=0))
    return e[classe] / e.sum(axis=1)


def resumo_junho(p, y, corte=0.5):
    y = np.asarray(y)
    marca = np.asarray(p) > corte
    return int((marca & (y == 1)).sum()), int((marca & (y == 0)).sum())


FAIXAS = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0000001)]


def faixas(p, y):
    """Quantas compras caem em cada faixa de P(fraude) e quantas delas foram fraude."""
    p, y = np.asarray(p), np.asarray(y)
    linhas = []
    for a, b in FAIXAS:
        dentro = (p >= a) & (p < b) if a > 0 else (p < b)
        linhas.append((a, b, int(dentro.sum()), int((dentro & (y == 1)).sum())))
    return linhas


def _nome_faixa(a, b):
    if a == 0:
        return "abaixo de 50%"
    if b > 1:
        return "99% ou mais"
    return f"de {br(100 * a, 0)}% a {br(100 * b, 0)}%"


# ---------------------------------------------------------------- a aula

def _parte(n, titulo):
    print(f"\n{n}. {titulo}")


def main():
    dados = carregar_mensagens()
    treino, junho = separar_mensagens(dados)
    modelo = contar(treino["texto"].map(palavras), treino["assunto"])
    total = len(treino)
    v = len(modelo["vocab"])
    tam = {a: sum(modelo["n_palavras"][a].values()) for a in ASSUNTOS}

    print(f"{total} mensagens de {treino['data'].min()} a {treino['data'].max()} e {len(junho)} de junho")
    print("vocabulário de janeiro a maio: %d palavras diferentes" % v)

    _parte(1, "As mensagens de janeiro a maio, por assunto")
    print("  assunto      mensagens   parte    palavras no total")
    for a in ASSUNTOS:
        n = modelo["n_msgs"][a]
        print(f"  {a:<11}{n:>10}  {pct(n / total):>6}  {tam[a]:>18}")

    _parte(2, f"À mão: «{MAO}», contando palavras (multinomial, alpha = 1)")
    ps = palavras(MAO)
    print("  palavras: " + ", ".join(ps))
    print("  vezes de cada palavra em cada assunto, de janeiro a maio:")
    print("  palavra    " + "".join(f"{a:>11}" for a in ASSUNTOS))
    for p in ps:
        print(f"  {p:<11}" + "".join(f"{modelo['n_palavras'][a][p]:>11}" for a in ASSUNTOS))
    print("  P(palavra | assunto) = (vezes + 1) ÷ (palavras do assunto + %d):" % v)
    produtos = {}
    for a in ASSUNTOS:
        prior = modelo["n_msgs"][a] / total
        qs = [(modelo["n_palavras"][a][p] + 1) / (tam[a] + v) for p in ps]
        produtos[a] = prior * math.prod(qs)
        contas = "  ".join(f"{p} {modelo['n_palavras'][a][p] + 1}/{tam[a] + v} = {br(q, 6)}" for p, q in zip(ps, qs))
        print(f"  {a:<10} P(assunto) {modelo['n_msgs'][a]}/{total} = {br(prior, 4)}  {contas}")
    z = sum(produtos.values())
    print("  produto P(assunto) × P(pix) × P(nao) × P(chegou), e a parte de cada assunto na soma:")
    for a in ASSUNTOS:
        print(f"  {a:<10} {produtos[a]:.4e}".replace(".", ",") + f"   {pct(produtos[a] / z)}")
    pm = probabilidades(multinomial_log(ps, modelo))
    print("  pelo multinomial_log: " + _linha_probs(pm))

    _parte(3, "Sem suavização (alpha = 0)")
    for texto in (MAO, MISTURA):
        try:
            p0 = probabilidades(multinomial_log(palavras(texto), modelo, alpha=0))
            print(f"  «{texto}»: " + _linha_probs(p0))
        except ValueError as erro:
            print(f"  «{texto}»: {erro}")
    zeros = {a: [p for p in palavras(MISTURA) if p in modelo["vocab"] and modelo["n_palavras"][a][p] == 0]
             for a in ASSUNTOS}
    for a in ASSUNTOS:
        print(f"  {a:<10} palavras da segunda mensagem que nunca apareceram no assunto: {', '.join(zeros[a]) or '(nenhuma)'}")

    _parte(4, f"À mão: «{MAO}» como presença (Bernoulli, alpha = 1), só com {len(MAO_VOCAB)} palavras e 2 assuntos")
    red = reduzir(modelo, MAO_VOCAB, MAO_ASSUNTOS)
    tem = set(ps)
    print("  vocabulário: " + ", ".join(MAO_VOCAB) + "   (a mensagem tem: " + ", ".join(p for p in MAO_VOCAB if p in tem) + ")")
    print("  P(aparece | assunto) = (mensagens com a palavra + 1) ÷ (mensagens do assunto + 2):")
    fat = {}
    sem_falta = {}
    for a in MAO_ASSUNTOS:
        n = red["n_msgs"][a]
        prior = n / sum(red["n_msgs"].values())
        termos, f, f2 = [], prior, prior
        for p in MAO_VOCAB:
            q = (red["n_com"][a][p] + 1) / (n + 2)
            if p in tem:
                termos.append(f"{p} {red['n_com'][a][p] + 1}/{n + 2} = {br(q, 4)}")
                f *= q
                f2 *= q
            else:
                termos.append(f"falta {p}: 1 − {red['n_com'][a][p] + 1}/{n + 2} = {br(1 - q, 4)}")
                f *= 1 - q
        fat[a], sem_falta[a] = f, f2
        print(f"  {a:<10} P(assunto) {n}/{sum(red['n_msgs'].values())} = {br(prior, 4)}  " + "  ".join(termos))
    zz, zz2 = sum(fat.values()), sum(sem_falta.values())
    print("  com a palavra que falta:  " + "  ".join(f"{a} {pct(fat[a] / zz)}" for a in MAO_ASSUNTOS))
    print("  sem a palavra que falta:  " + "  ".join(f"{a} {pct(sem_falta[a] / zz2)}" for a in MAO_ASSUNTOS))
    pr = probabilidades(bernoulli_log(ps, red))
    print("  pelo bernoulli_log, com as 4 palavras: " + "  ".join(f"{a} {pct(pr[a])}" for a in MAO_ASSUNTOS))
    pb = probabilidades(bernoulli_log(ps, modelo))
    print(f"  com as {v} palavras e os 4 assuntos: " + _linha_probs(pb))

    _parte(5, f"Uma mensagem com dois assuntos: «{MISTURA}»")
    pm = probabilidades(multinomial_log(palavras(MISTURA), modelo))
    pb = probabilidades(bernoulli_log(palavras(MISTURA), modelo))
    print("  multinomial: " + _linha_probs(pm) + f"   -> {_melhor(pm)}")
    print("  Bernoulli:   " + _linha_probs(pb) + f"   -> {_melhor(pb)}")
    tem = set(palavras(MISTURA))
    contrib = []
    for p in modelo["vocab"]:
        if p in tem:
            continue
        qa = (modelo["n_com"]["atraso"][p] + 1) / (modelo["n_msgs"]["atraso"] + 2)
        qt = (modelo["n_com"]["troca"][p] + 1) / (modelo["n_msgs"]["troca"] + 2)
        contrib.append(((1 - qa) / (1 - qt), p))
    contrib.sort()
    soma = math.prod(c for c, _ in contrib)
    print(f"  no Bernoulli, as {len(contrib)} palavras que faltam multiplicam a chance de atraso contra troca por {br(soma, 2)}")
    print("  as três que mais pesam (mensagens com a palavra: atraso de %d, troca de %d):"
          % (modelo["n_msgs"]["atraso"], modelo["n_msgs"]["troca"]))
    for c, p in contrib[:3]:
        print(f"    falta {p:<8} atraso {modelo['n_com']['atraso'][p]:>4}  troca {modelo['n_com']['troca'][p]:>4}   multiplica por {br(c, 2)}")
    iguais = sum(_melhor(probabilidades(multinomial_log(palavras(t), modelo)))
                 == _melhor(probabilidades(bernoulli_log(palavras(t), modelo))) for t in junho["texto"])
    print(f"  nas {len(junho)} mensagens de junho, as duas variantes dão o mesmo assunto em {iguais}")

    _parte(6, f"A mesma frase repetida: «{REPETIDA}»")
    print("  vezes   multinomial (atraso, troca)   Bernoulli (atraso, troca)")
    for k in (1, 2, 3):
        t = ". ".join([REPETIDA] * k)
        pm = probabilidades(multinomial_log(palavras(t), modelo))
        pb = probabilidades(bernoulli_log(palavras(t), modelo))
        print(f"  {k:>5}   {pct(pm['atraso']):>12}  {pct(pm['troca']):>12}   {pct(pb['atraso']):>10}  {pct(pb['troca']):>10}")

    compras = carregar_compras()
    conh, novas = separar_compras(compras)
    print(f"\n{len(conh)} compras de janeiro a maio ({int(conh['fraude'].sum())} fraudes) "
          f"e {len(novas)} de junho ({int(novas['fraude'].sum())} fraudes)")

    _parte(7, "Compras com a curva normal nas colunas como vieram (gaussiana)")
    m = ajustar_gaussiana(conh[COLUNAS], conh["fraude"])
    print("  coluna     legítima: média     desvio   fraude: média     desvio")
    for c in COLUNAS:
        print(f"  {CURTO[c]:<10}{br(m[0]['media'][c]):>16}{br(math.sqrt(m[0]['var'][c])):>11}"
              f"{br(m[1]['media'][c]):>16}{br(math.sqrt(m[1]['var'][c])):>11}")
    leg = conh[conh["fraude"] == 0]
    mu, dp = m[0]["media"]["valor"], math.sqrt(m[0]["var"]["valor"])
    abaixo = 0.5 * (1 + math.erf((0 - mu) / (dp * math.sqrt(2))))
    print(f"  valor das legítimas: metade abaixo de {br(float(leg['valor'].median()))}; a curva normal com essa média"
          f" e esse desvio põe {pct(abaixo)} delas abaixo de R$ 0")
    p = gaussiana_prob(novas[COLUNAS], m)
    f1, l1 = resumo_junho(p, novas["fraude"])
    print(f"  junho, P(fraude) acima de 50%: marca {f1} das {int(novas['fraude'].sum())} fraudes e {l1} compras legítimas")
    print("  P(fraude) que o modelo deu     compras   fraudes")
    for a, b, n, f in faixas(p, novas["fraude"]):
        print(f"  {_nome_faixa(a, b):<28}{n:>9}{f:>10}")

    _parte(8, "As mesmas compras com valor, conta e distância em log")
    m = ajustar_gaussiana(em_log(conh), conh["fraude"])
    p = gaussiana_prob(em_log(novas), m)
    f1, l1 = resumo_junho(p, novas["fraude"])
    print(f"  junho, P(fraude) acima de 50%: marca {f1} das {int(novas['fraude'].sum())} fraudes e {l1} compras legítimas")
    print("  P(fraude) que o modelo deu     compras   fraudes")
    for a, b, n, f in faixas(p, novas["fraude"]):
        print(f"  {_nome_faixa(a, b):<28}{n:>9}{f:>10}")

    _parte(9, "Dentro de cada classe, as colunas andam juntas? (correlação, com as colunas em log)")
    x = em_log(conh)
    pares = [("idade_conta_dias", "distancia_entrega_km"), ("idade_conta_dias", "tentativas_cartao"),
             ("distancia_entrega_km", "tentativas_cartao")]
    print("  par                  nas legítimas   nas fraudes")
    for a, b in pares:
        cl = float(x[conh["fraude"] == 0][a].corr(x[conh["fraude"] == 0][b]))
        cf = float(x[conh["fraude"] == 1][a].corr(x[conh["fraude"] == 1][b]))
        print(f"  {CURTO[a] + ' e ' + CURTO[b]:<20}{br(cl):>15}{br(cf):>14}")
    fr = conh[conh["fraude"] == 1]
    nova = fr["idade_conta_dias"] <= 20
    print(f"  fraudes com conta de 20 dias ou menos: {int(nova.sum())}; mediana da entrega "
          f"{br(float(fr[nova]['distancia_entrega_km'].median()), 1)} km e {int(fr[nova]['tentativas_cartao'].median())} tentativas")
    print(f"  as outras {int((~nova).sum())}: mediana da entrega {br(float(fr[~nova]['distancia_entrega_km'].median()), 1)} km"
          f" e {int(fr[~nova]['tentativas_cartao'].median())} tentativas")

    _parte(10, "O canal como categoria (app ou site, alpha = 1)")
    cats = sorted(conh["canal"].unique())
    print("  canal   " + "".join(f"{'legítima':>16}{'fraude':>16}"))
    for c in cats:
        cel = []
        for k in (0, 1):
            parte = conh[conh["fraude"] == k]
            n_c = int((parte["canal"] == c).sum())
            cel.append(f"{n_c + 1}/{len(parte) + len(cats)} = {br((n_c + 1) / (len(parte) + len(cats)), 3)}")
        print(f"  {c:<8}" + "".join(f"{t:>16}" for t in cel))


if __name__ == "__main__":
    main()
