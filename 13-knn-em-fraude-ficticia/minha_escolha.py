"""O exercício: escolher o K e o corte do voto com a análise a R$ 40 e abrir junho uma vez.

Rode de dentro da pasta da aula:

    python3 testar_minha_escolha.py      # testa as duas funções que você escreve aqui
    python3 minha_escolha.py             # escolhe nas dobras, mede o tempo e grava meu-benchmark.json
    python3 minha_escolha.py --abrir-teste

Escreva as duas funções abaixo no próprio arquivo, sem importar as do benchmark.py. O resto (as
compras, o KNN, a grade, o registro e a abertura do teste) vem do benchmark.py da aula.
"""
import sys

import numpy as np

from benchmark import AQUI, rodar

PRECO_DO_DIEGO = 40.0


def dobras_estratificadas(fraude, n_dobras, sorteio):
    """A dobra (de 0 a n_dobras − 1) de cada compra, num array de inteiros do tamanho de fraude.

    A regra é a da aula: crie UM gerador, np.random.default_rng(sorteio). Com ele, embaralhe primeiro
    as posições das compras legítimas (np.flatnonzero(fraude == 0)) e depois as das fraudes
    (np.flatnonzero(fraude == 1)), com gerador.permutation. Em cada grupo, a posição que sai em
    i-ésimo lugar fica na dobra i % n_dobras.
    Pare com ValueError se algum rótulo não for 0 ou 1, ou se n_dobras for menor que 2 ou maior que
    o número de fraudes. Não mude o fraude que recebeu.
    """
    raise NotImplementedError("escreva dobras_estratificadas")


def custo_da_regra(votos_k, fraude, valor, corte, preco):
    """Um dicionário com marcadas, pegas, alarmes, passou e custo, para as compras de uma dobra.

    votos_k: quantos dos K vizinhos de cada compra foram fraude; fraude: o rótulo de cada compra;
    valor: o valor de cada compra, em reais. Uma compra é marcada quando votos_k >= corte.
    pegas: fraudes marcadas; alarmes: compras legítimas marcadas; passou: soma do valor das fraudes
    que NÃO foram marcadas; custo: preco × marcadas + passou.
    Pare com ValueError se os três tiverem tamanhos diferentes, se o corte for menor que 1 ou se o
    preço for negativo.
    """
    raise NotImplementedError("escreva custo_da_regra")


if __name__ == "__main__":
    sys.exit(rodar(sys.argv[1:], registro=AQUI / "meu-benchmark.json", preco=PRECO_DO_DIEGO,
                   dividir=dobras_estratificadas, custo=custo_da_regra))
