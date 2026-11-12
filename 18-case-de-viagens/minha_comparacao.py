"""Exercício da aula «Case de viagens»: a ligação terceirizada e a equipe maior.

Escreva as duas funções abaixo. Depois rode, de dentro da pasta da aula:

    python3 testar_minha_comparacao.py
    python3 minha_comparacao.py

Antes de você escrever as funções, os dois param com NotImplementedError.
"""
import numpy as np
import pandas as pd

from case_viagens import CANDIDATOS, FAIXAS, LISTA_DE_HOJE, TRIMESTRES, br, ler, nota_do_candidato, reais

PRECO_NOVO = 35.0             # a ligação da empresa terceirizada
VALOR_ALCANCADO = 100.0       # o mesmo da aula: 1 em 3 remarca, R$ 300 de comissão
FRACOES = [(1, 10), (1, 5), (1, 3)]     # para quantas reservas ligar; 1 em 3 é o teto novo
VANTAGEM_MINIMA = 1000.0
EMPATE = 500.0


def treino_da_dobra(base, inicio):
    """As reservas de `base` cujo desfecho já existia em `inicio`.

    Entram as feitas antes de `inicio` e com a viagem antes de `inicio` (a viagem no próprio dia fica
    fora). `inicio` é uma data em texto ('2025-07-01') ou um pd.Timestamp. Retorna um DataFrame novo,
    na ordem de `base`. Para com ValueError se faltar a coluna data_reserva ou data_viagem, se faltar
    alguma data ou se alguma viagem for antes da própria reserva. Não muda o que recebeu.
    """
    raise NotImplementedError("escreva treino_da_dobra(base, inicio)")


def economia_da_lista(nota, y, ids, quantos, preco, valor):
    """Liga para as `quantos` reservas de nota mais alta e retorna (canceladas na lista, economia).

    `nota`, `y` e `ids` são listas ou arrays do mesmo tamanho; `y` vale 0 ou 1; `ids` é o número da
    reserva. Em empate de nota, vem antes a reserva de número menor. economia = valor × canceladas na
    lista − preco × quantos. Para com ValueError se os tamanhos forem diferentes, se `quantos` for menor
    que 1 ou maior que o número de reservas, se faltar nota, se `y` tiver algo além de 0 e 1 ou se
    `ids` tiver número repetido. Não muda o que recebeu.
    """
    raise NotImplementedError("escreva economia_da_lista(nota, y, ids, quantos, preco, valor)")


# ---------- daqui para baixo já está pronto ----------

def dobras_novas(base):
    saida = []
    for nome, ini, fim in TRIMESTRES:
        treino = treino_da_dobra(base, ini)
        medidas = base[(base["data_reserva"] >= pd.Timestamp(ini)) & (base["data_reserva"] < pd.Timestamp(fim))]
        saida.append((nome, treino, medidas))
    return saida


def parte_a(ds):
    print(f"A. Os candidatos com a ligação a {reais(PRECO_NOVO)}: economia somada nas três dobras, por tamanho de lista")
    print(f"  {'modelo':<36}" + "".join(f"{'1 em ' + str(b):>11}" for _, b in FRACOES) + f"{'melhor':>10}")
    res, notas = {}, {}
    for nome in CANDIDATOS:
        notas[nome] = [nota_do_candidato(nome, tr, me) for _, tr, me in ds]
        somas = []
        for a, b in FRACOES:
            s = 0.0
            for nota, (_, _, me) in zip(notas[nome], ds):
                _, e = economia_da_lista(nota, me["cancelou"], me["id_reserva"], len(me) * a // b,
                                         PRECO_NOVO, VALOR_ALCANCADO)
                s += e
            somas.append(s)
        k = int(np.argmax(somas))
        res[nome] = (somas[k], FRACOES[k])
        print(f"  {nome:<36}" + "".join(f"{reais(s):>11}" for s in somas) + f"{'1 em ' + str(FRACOES[k][1]):>10}")
    print(f"  em cada linha, o melhor tamanho é o de maior economia somada; o teto novo é 1 em 3")
    print()
    return res, notas


def parte_b(res):
    print("B. A escolha, pela regra da aula")
    modelos = CANDIDATOS[1:]
    melhor = max(res[n][0] for n in modelos)
    empatados = [n for n in modelos if melhor - res[n][0] <= EMPATE]
    escolhido = empatados[0]
    hoje = res[LISTA_DE_HOJE][0]
    print("  empatados com o melhor: " + "; ".join(empatados))
    print(f"  escolhido: {escolhido}, ligando para 1 em {res[escolhido][1][1]}, {reais(res[escolhido][0])}")
    print(f"  lista de hoje, no melhor tamanho dela (1 em {res[LISTA_DE_HOJE][1][1]}): {reais(hoje)}; "
          f"diferença {reais(res[escolhido][0] - hoje)}")
    troca = res[escolhido][0] - hoje >= VANTAGEM_MINIMA
    fica = escolhido if troca else LISTA_DE_HOJE
    print(f"  fica {'o modelo: ' + escolhido if troca else 'a lista de hoje'}")
    print()
    return fica


def parte_c(ds, nota_fica, fracao):
    a, b = fracao
    print(f"C. Onde a lista de 1 em {b} erra, somando as três dobras")
    print(f"  {'antecedência':<17}{'tarifa':<11}{'reservas':>9}{'canceladas':>12}{'ligações':>10}"
          f"{'pegas':>7}{'perdidas':>10}{'ligações à toa':>16}")
    tabela = {}
    for nota, (_, _, me) in zip(nota_fica, ds):
        q = len(me) * a // b
        ordem = np.lexsort((me["id_reserva"].to_numpy(), -np.asarray(nota, float)))
        na_lista = np.zeros(len(me), dtype=bool)
        na_lista[ordem[:q]] = True
        y = me["cancelou"].to_numpy()
        for lo, hi, rotulo in FAIXAS:
            for flex in (0, 1):
                g = (me["antecedencia_dias"].between(lo, hi) & (me["tarifa_flexivel"] == flex)).to_numpy()
                t = tabela.setdefault((rotulo, flex), [0, 0, 0, 0])
                t[0] += int(g.sum())
                t[1] += int(y[g].sum())
                t[2] += int(na_lista[g].sum())
                t[3] += int((y[g] == 1)[na_lista[g]].sum())
    tot = [0, 0, 0, 0]
    for (rotulo, flex), (n, canc, lig, peg) in tabela.items():
        tot = [u + v for u, v in zip(tot, (n, canc, lig, peg))]
        print(f"  {rotulo:<17}{'flexível' if flex else 'comum':<11}{n:>9}{canc:>12}{lig:>10}{peg:>7}"
              f"{canc - peg:>10}{lig - peg:>16}")
    print(f"  {'total':<28}{tot[0]:>9}{tot[1]:>12}{tot[2]:>10}{tot[3]:>7}{tot[1] - tot[3]:>10}{tot[2] - tot[3]:>16}")


def main():
    base = ler()
    ds = dobras_novas(base)
    res, notas = parte_a(ds)
    fica = parte_b(res)
    parte_c(ds, notas[fica], res[fica][1])


if __name__ == "__main__":
    main()
