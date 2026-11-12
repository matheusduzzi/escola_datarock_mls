"""Exercício da aula «Naive Bayes na prática»: ler os erros de julho.

Escreva as duas funções abaixo, sem importar nada que já faça a conta. Depois rode, de dentro
da pasta da aula:

    python3 testar_minha_analise.py
    python3 minha_analise.py

Antes de você escrever as funções, os dois param com NotImplementedError. É o esperado.
"""
from filas import ASSUNTOS, carregar, carregar_julho, palavras, pct, prever, separar, treinar


def matriz_confusao(reais, previstos, assuntos=ASSUNTOS):
    """Quantas mensagens de cada assunto marcado foram para cada assunto previsto.

    reais: o assunto marcado de cada mensagem. previstos: o que o modelo disse, na mesma ordem.
    Retorna um dicionário de dicionários: matriz[marcado][previsto] = quantas mensagens.
    As linhas e as colunas são todos os assuntos de `assuntos`, nessa ordem, com 0 onde não houve
    mensagem. Para com ValueError se reais e previstos tiverem tamanhos diferentes ou se aparecer um
    assunto fora de `assuntos`. Não muda o que recebeu.
    """
    raise NotImplementedError("escreva matriz_confusao(reais, previstos, assuntos)")


def cobertura(ps, vocab):
    """A parte das palavras da mensagem que está no vocabulário, de 0 a 1.

    ps: a lista de palavras da mensagem, já passada por palavras(); a palavra repetida conta cada vez.
    vocab: as palavras que o modelo conhece (lista ou conjunto). Mensagem sem palavra dá 0.0.
    Para com TypeError se ps for um texto em vez de uma lista.
    """
    raise NotImplementedError("escreva cobertura(ps, vocab)")


# ----------------------------------------------------------------------------- o que roda julho
# Daqui para baixo, nada para mudar.

CORTE_PROB = 0.90       # a proposta do Diego
CORTE_COBERTURA = 0.5   # menos da metade das palavras conhecidas


def main():
    treino, _ = separar(carregar())
    modelo = treinar(treino)
    julho = carregar_julho()
    linhas = []
    for l in julho:
        ps = palavras(l["texto"])
        nb, pr = prever(ps, modelo)
        linhas.append({"id": l["id"], "texto": l["texto"], "marcado": l["assunto"], "previsto": nb,
                       "prob": pr[nb], "ps": ps, "novas": [p for p in ps if p not in set(modelo["vocab"])]})

    print(f"{len(treino)} mensagens de janeiro a maio no treino e {len(julho)} de julho")
    print("\nA. Matriz do Naive Bayes em julho (linha: o que o Diego marcou; coluna: o que o modelo disse)")
    m = matriz_confusao([l["marcado"] for l in linhas], [l["previsto"] for l in linhas])
    for l in linhas:
        l["cob"] = cobertura(l["ps"], modelo["vocab"])
    print("  " + " " * 11 + "".join(f"{a:>11}" for a in ASSUNTOS) + "      total")
    for a in ASSUNTOS:
        print(f"  {a:<11}" + "".join(f"{m[a][b]:>11}" for b in ASSUNTOS) + f"{sum(m[a].values()):>11}")
    print("  " + "recebeu".ljust(11) + "".join(f"{sum(m[r][b] for r in ASSUNTOS):>11}" for b in ASSUNTOS))
    acertos = sum(m[a][a] for a in ASSUNTOS)
    print(f"  acertos: {acertos} de {len(linhas)}")

    print("\nB. Os erros, com a probabilidade e a parte das palavras que o modelo conhece")
    print("  id  marcado    disse      prob.   conhece  palavras fora do vocabulário")
    for l in linhas:
        if l["previsto"] != l["marcado"]:
            print(f"  {l['id']:>2}  {l['marcado']:<10} {l['previsto']:<10}{pct(l['prob'], 1):>6}   {pct(l['cob'], 1):>6}  "
                  f"{', '.join(l['novas'])}")
            print(f"      «{l['texto']}»")

    print("\nC. Dois sinais para mandar a mensagem para uma pessoa")
    sinais = [(f"probabilidade da fila abaixo de {pct(CORTE_PROB, 1)}", lambda l: l["prob"] < CORTE_PROB),
              (f"menos de {pct(CORTE_COBERTURA, 1)} das palavras conhecidas", lambda l: l["cob"] < CORTE_COBERTURA)]
    erros = sum(l["previsto"] != l["marcado"] for l in linhas)
    for nome, sinal in sinais:
        sep = [l for l in linhas if sinal(l)]
        e = sum(l["previsto"] != l["marcado"] for l in sep)
        print(f"  {nome}: separa {len(sep)} mensagens, {e} erros e {len(sep) - e} acertos; "
              f"passam {erros - e} dos {erros} erros")


if __name__ == "__main__":
    main()
