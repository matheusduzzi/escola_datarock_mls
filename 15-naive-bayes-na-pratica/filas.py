"""Regra de palavras contra Naive Bayes nas mensagens do atendimento da Loja Exemplo.

Só a biblioteca padrão do Python (3.9 ou mais novo). Rode de dentro da pasta da aula:

    python3 filas.py

Também serve de módulo para o exercício:

    from filas import (ASSUNTOS, palavras, carregar, separar, treinar, prever, regra,
                       fila_maior, carregar_julho, onde_aparece)

As mensagens de janeiro a maio são as que o modelo conhece, com o assunto marcado. Junho fica
fechado até a parte 4. As mensagens de julho são as que o Diego separou e marcou. Não há sorteio:
a saída e o arquivo de previsões saem iguais toda vez.
"""
import csv
import hashlib
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
MENSAGENS = AQUI.parent / "dados" / "mensagens" / "mensagens.csv"
JULHO = AQUI / "julho.csv"
SAIDA = AQUI / "saida" / "previsoes.csv"
ASSUNTOS = ["atraso", "troca", "pagamento", "outro"]
MESES = {"01": "janeiro", "02": "fevereiro", "03": "março", "04": "abril", "05": "maio", "06": "junho"}

# A regra que o atendimento escreveu antes do modelo. A ordem conta: a primeira lista que
# tiver uma palavra da mensagem decide. Sem palavra de nenhuma lista, a mensagem vai para outro.
REGRA = [
    ("pagamento", ["boleto", "cartao", "cobraram", "estorno", "fatura", "pagamento", "pagar", "paguei", "pix"]),
    ("troca", ["defeito", "devolver", "diferente", "errada", "errado", "quebrado", "reembolso", "troca", "trocar"]),
    ("atraso", ["atrasada", "atrasado", "cade", "chega", "chegar", "chegou", "entrega", "prazo", "rastreio",
                "transportadora"]),
]


def br(x, casas=1):
    """Número com vírgula decimal, como na aula."""
    return f"{x:.{casas}f}".replace(".", ",")


def pct(a, b):
    return br(100 * a / b) + "%"


def palavras(texto):
    """Minúsculas, sem acento, só letras: «Não CHEGOU! pedido 123» vira ['nao', 'chegou', 'pedido']."""
    if not isinstance(texto, str):
        raise TypeError("palavras recebe um texto")
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z]+", t)


def _ler(caminho, colunas):
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com dados/ na raiz do kit.")
    with open(caminho, encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f))
    faltam = [c for c in colunas if linhas and c not in linhas[0]]
    if not linhas or faltam:
        raise ValueError(f"{caminho.name} sem linhas ou sem as colunas {', '.join(faltam or colunas)}")
    fora = sorted({l["assunto"] for l in linhas} - set(ASSUNTOS))
    if fora:
        raise ValueError(f"{caminho.name} tem assunto fora da lista: {', '.join(fora)}")
    return linhas


def carregar(caminho=MENSAGENS):
    return _ler(caminho, ["id_mensagem", "data", "texto", "assunto"])


def carregar_julho(caminho=JULHO):
    return _ler(caminho, ["id", "data", "texto", "assunto"])


def separar(linhas):
    """Janeiro a maio (o que o modelo conhece) e junho (o teste, aberto uma vez)."""
    junho = [l for l in linhas if l["data"].startswith("2026-06-")]
    return [l for l in linhas if not l["data"].startswith("2026-06-")], junho


def treinar(linhas):
    """O Naive Bayes multinomial da aula 14 guarda só contagens. Vocabulário só das linhas recebidas."""
    n_msgs = Counter()
    vezes = {a: Counter() for a in ASSUNTOS}
    for l in linhas:
        n_msgs[l["assunto"]] += 1
        vezes[l["assunto"]].update(palavras(l["texto"]))
    vocab = sorted({p for a in ASSUNTOS for p in vezes[a]})
    return {"n_msgs": n_msgs, "vezes": vezes, "vocab": vocab, "tam": {a: sum(vezes[a].values()) for a in ASSUNTOS}}


def prever(ps, modelo, alpha=1.0):
    """(assunto escolhido, {assunto: probabilidade}). Palavra fora do vocabulário fica de fora.

    Empate exato fica com o assunto que vem antes em ASSUNTOS.
    """
    if isinstance(ps, str):
        raise TypeError("passe a lista de palavras, não o texto: use palavras(texto)")
    total = sum(modelo["n_msgs"].values())
    v = len(modelo["vocab"])
    vocab = set(modelo["vocab"])
    logs = {}
    for a in ASSUNTOS:
        if modelo["n_msgs"][a] == 0:
            logs[a] = -math.inf
            continue
        s = math.log(modelo["n_msgs"][a] / total)
        for p in ps:
            if p in vocab:
                s += math.log((modelo["vezes"][a][p] + alpha) / (modelo["tam"][a] + alpha * v))
        logs[a] = s
    mx = max(logs.values())
    soma = sum(math.exp(x - mx) for x in logs.values())
    probs = {a: math.exp(logs[a] - mx) / soma for a in ASSUNTOS}
    escolhido = max(ASSUNTOS, key=lambda a: (probs[a], -ASSUNTOS.index(a)))
    return escolhido, probs


def regra(ps, listas=None):
    """A regra de palavras: (assunto, palavra que decidiu ou None). listas troca a REGRA, para testar consertos."""
    tem = set(ps)
    for assunto, lista in (REGRA if listas is None else listas):
        for p in lista:
            if p in tem:
                return assunto, p
    return "outro", None


def onde_aparece(palavra, linhas):
    """Em quantas mensagens de cada assunto a palavra aparece, uma vez por mensagem."""
    return {a: sum(1 for l in linhas if l["assunto"] == a and palavra in set(palavras(l["texto"]))) for a in ASSUNTOS}


def fila_maior(linhas):
    """O chute mais simples: a fila com mais mensagens nas linhas conhecidas."""
    c = Counter(l["assunto"] for l in linhas)
    return max(ASSUNTOS, key=lambda a: (c[a], -ASSUNTOS.index(a)))


def _parte(n, titulo):
    print(f"\n{n}. {titulo}")


def _linha(nome, acertos, total):
    return f"  {nome:<14}{acertos:>4} de {total}   {pct(acertos, total):>6}"


def gravar_previsoes(linhas_por_mes, modelo, caminho=SAIDA):
    """Grava uma linha por mensagem com as três previsões e devolve a impressão digital do arquivo."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    chute = fila_maior_modelo(modelo)
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["mes", "id", "assunto", "fila_maior", "regra", "naive_bayes", "prob_naive_bayes"])
        for mes, linhas, chave in linhas_por_mes:
            for l in linhas:
                ps = palavras(l["texto"])
                nb, pr = prever(ps, modelo)
                w.writerow([mes, l[chave], l["assunto"], chute, regra(ps)[0], nb, f"{pr[nb]:.4f}"])
    return hashlib.sha256(caminho.read_bytes()).hexdigest()[:12]


def fila_maior_modelo(modelo):
    return max(ASSUNTOS, key=lambda a: (modelo["n_msgs"][a], -ASSUNTOS.index(a)))


def main():
    todas = carregar()
    treino, junho = separar(todas)
    modelo = treinar(treino)
    vocab = set(modelo["vocab"])
    print(f"{len(treino)} mensagens de janeiro a maio e {len(junho)} de junho")
    print(f"vocabulário de janeiro a maio: {len(vocab)} palavras")

    _parte(1, "Os dois pontos de partida")
    chute = fila_maior(treino)
    print(f"  fila maior: toda mensagem vai para {chute} ({modelo['n_msgs'][chute]} de {len(treino)} em janeiro a maio)")
    print("  regra de palavras, nessa ordem (a primeira lista com uma palavra da mensagem decide):")
    for assunto, lista in REGRA:
        print(f"    {assunto:<10} se tiver: {', '.join(lista)}")
    print("    outro      se não tiver nenhuma")

    _parte(2, "Escolher antes de junho: cada mês, com os meses anteriores como treino")
    print("  mês          mensagens   fila maior        regra   Naive Bayes")
    soma = Counter()
    for mm in ["02", "03", "04", "05"]:
        antes = [l for l in treino if l["data"][5:7] < mm]
        mes = [l for l in treino if l["data"][5:7] == mm]
        m = treinar(antes)
        c = fila_maior(antes)
        ac = {"fila maior": sum(l["assunto"] == c for l in mes),
              "regra": sum(regra(palavras(l["texto"]))[0] == l["assunto"] for l in mes),
              "Naive Bayes": sum(prever(palavras(l["texto"]), m)[0] == l["assunto"] for l in mes)}
        soma.update(ac)
        soma["n"] += len(mes)
        print(f"  {MESES[mm]:<12}{len(mes):>9}   {pct(ac['fila maior'], len(mes)):>10}   "
              f"{pct(ac['regra'], len(mes)):>10}   {pct(ac['Naive Bayes'], len(mes)):>11}")
    print(f"  {'os quatro':<12}{soma['n']:>9}   {pct(soma['fila maior'], soma['n']):>10}   "
          f"{pct(soma['regra'], soma['n']):>10}   {pct(soma['Naive Bayes'], soma['n']):>11}")
    erros_regra_treino = [l for l in treino if regra(palavras(l["texto"]))[0] != l["assunto"]]
    print(f"  a regra erra {len(erros_regra_treino)} das {len(treino)} mensagens de janeiro a maio")

    _parte(3, "Os erros da regra de janeiro a maio, por par (marcado -> regra)")
    pares = Counter((l["assunto"], regra(palavras(l["texto"]))[0]) for l in erros_regra_treino)
    for (real, prev), n in sorted(pares.items(), key=lambda x: (-x[1], ASSUNTOS.index(x[0][0]))):
        exemplo = next(l for l in erros_regra_treino if (l["assunto"], regra(palavras(l["texto"]))[0]) == (real, prev))
        decidiu = regra(palavras(exemplo["texto"]))[1]
        motivo = f"decidiu por «{decidiu}»" if decidiu else "nenhuma palavra das listas"
        print(f"  {real:<10}-> {prev:<10}{n:>4}   {motivo:<28} ex.: {exemplo['texto']}")
    antes = [regra(palavras(l["texto"]))[0] for l in treino]
    for n_c, (fila, extra) in enumerate([("atraso", ["comprei", "nada"]), ("pagamento", ["cupom", "desconto"])], start=1):
        listas = [(a, lista + extra if a == fila else lista) for a, lista in REGRA]
        depois = [regra(palavras(l["texto"]), listas)[0] for l in treino]
        melhor = sum(d == l["assunto"] != a for a, d, l in zip(antes, depois, treino))
        pior = sum(a == l["assunto"] != d for a, d, l in zip(antes, depois, treino))
        acerta = sum(d == l["assunto"] for d, l in zip(depois, treino))
        print(f"  conserto {n_c}: {' e '.join('«' + e + '»' for e in extra)} na lista de {fila}: acerta {acerta} de "
              f"{len(treino)} (antes {len(treino) - len(erros_regra_treino)}); {melhor} melhoram, {pior} pioram")
        if pior:
            ex = min((l for a, d, l in zip(antes, depois, treino) if a == l["assunto"] != d), key=lambda l: (len(l["texto"]), l["texto"]))
            print(f"    piora, por exemplo: «{ex['texto']}», marcado {ex['assunto']}, vai para {fila}")

    _parte(4, "Junho, uma vez, com o treino de janeiro a maio")
    chute_j = sum(l["assunto"] == chute for l in junho)
    prev_regra = [regra(palavras(l["texto"]))[0] for l in junho]
    prev_nb = [prever(palavras(l["texto"]), modelo) for l in junho]
    ac_regra = sum(p == l["assunto"] for p, l in zip(prev_regra, junho))
    ac_nb = sum(p[0] == l["assunto"] for p, l in zip(prev_nb, junho))
    print(_linha("fila maior", chute_j, len(junho)))
    print(_linha("regra", ac_regra, len(junho)))
    print(_linha("Naive Bayes", ac_nb, len(junho)))
    pares_j = Counter((l["assunto"], p) for p, l in zip(prev_regra, junho) if p != l["assunto"])
    print("  erros da regra em junho, por par: " +
          ", ".join(f"{r} -> {p} {n}" for (r, p), n in sorted(pares_j.items(), key=lambda x: (-x[1], x[0]))))

    _parte(5, "Três mensagens de junho que a regra errou e o Naive Bayes acertou")
    for alvo in ["paguei no pix", "nao quero trocar nada", "comprei faz"]:
        l = next(x for x, p in zip(junho, prev_regra)
                 if " ".join(palavras(x["texto"])).find(alvo) >= 0 and p != x["assunto"])
        ps = palavras(l["texto"])
        nb, pr = prever(ps, modelo)
        print(f"  «{l['texto']}»")
        print(f"    marcado {l['assunto']}, regra {regra(ps)[0]}, Naive Bayes {nb} {pct(pr[nb], 1)}")
    for desc, quer in [("«paguei» e «chegou»", {"paguei", "chegou"}), ("«quero», «trocar» e «chega»", {"quero", "trocar", "chega"}),
                       ("«comprei» e «nada»", {"comprei", "nada"})]:
        c = Counter(l["assunto"] for l in treino if quer <= set(palavras(l["texto"])))
        print(f"  de janeiro a maio, mensagens com {desc}: " + ", ".join(f"{a} {c[a]}" for a in ASSUNTOS if c[a]))

    _parte(6, "Por que desconfiar dos 100%")
    vistas = {" ".join(palavras(l["texto"])) for l in treino}
    iguais = sum(" ".join(palavras(l["texto"])) in vistas for l in junho)
    novas = sum(any(p not in vocab for p in palavras(l["texto"])) for l in junho)
    menor = min(p[1][p[0]] for p in prev_nb)
    print(f"  mensagens de junho iguais, palavra por palavra, a uma de janeiro a maio: {iguais} de {len(junho)}")
    print(f"  mensagens de junho com alguma palavra fora do vocabulário: {novas} de {len(junho)}")
    print(f"  menor probabilidade da fila escolhida em junho: {pct(menor, 1)}")

    _parte(7, "Julho: as mensagens que o Diego separou e marcou")
    julho = carregar_julho()
    ac = {"fila maior": sum(l["assunto"] == chute for l in julho),
          "regra": sum(regra(palavras(l["texto"]))[0] == l["assunto"] for l in julho),
          "Naive Bayes": sum(prever(palavras(l["texto"]), modelo)[0] == l["assunto"] for l in julho)}
    print(f"  {len(julho)} mensagens: " + ", ".join(f"{a} {sum(l['assunto'] == a for l in julho)}" for a in ASSUNTOS))
    for nome in ["fila maior", "regra", "Naive Bayes"]:
        print(_linha(nome, ac[nome], len(julho)))
    com_nova = sum(any(p not in vocab for p in palavras(l["texto"])) for l in julho)
    print(f"  mensagens de julho com alguma palavra fora do vocabulário: {com_nova} de {len(julho)}")
    print("  a leitura dos erros de julho é o exercício (minha_analise.py)")

    _parte(8, "Previsões gravadas")
    digital = gravar_previsoes([("junho", junho, "id_mensagem"), ("julho", julho, "id")], modelo)
    print(f"  saida/previsoes.csv: {len(junho) + len(julho)} linhas, impressão digital {digital}")


if __name__ == "__main__":
    main()
