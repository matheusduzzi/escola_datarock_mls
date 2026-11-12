"""O exercício: escolher o alfa do Lasso nas mesmas dobras e abrir junho uma vez.

Rode de dentro da pasta da aula:

    python3 testar_minha_escolha.py      # testa as duas funções que você escreve aqui
    python3 minha_escolha.py             # escolhe o alfa do Lasso e grava meu-experimento.json
    python3 minha_escolha.py --abrir-teste

Escreva as duas funções abaixo no próprio arquivo, sem importar as do experimento.py. O resto (os
pedidos, as colunas, o ajuste, o registro e a abertura do teste) vem do experimento.py da aula.
"""
import sys

import numpy as np

from experimento import AQUI, ajustar, mse, padronizar, rodar

GRADE_LASSO = [0, 0.5, 1, 2, 5, 10, 20, 40]


def dividir_em_dobras(n, k, sorteio):
    """A dobra (de 0 a k − 1) de cada um dos n pedidos, num array de inteiros de tamanho n.

    A regra é a da aula: embaralhe as posições 0 a n − 1 com np.random.default_rng(sorteio).permutation(n).
    A posição que sai em i-ésimo lugar fica na dobra i % k.
    Pare com ValueError se k for menor que 2 ou maior que n.
    """
    raise NotImplementedError("escreva dividir_em_dobras")


def erro_nas_dobras(x, y, dobras, l1=0.0, l2=0.0):
    """Uma lista com o MSE de cada dobra, da dobra 0 à dobra k − 1.

    x: DataFrame com as colunas; y: os valores; dobras: a dobra de cada linha.
    Para cada dobra j: padronize com a média e o desvio das linhas que NÃO são da dobra j (use padronizar),
    ajuste nessas linhas (use ajustar, com l1 e l2) e meça o MSE nas linhas da dobra j (use mse).
    Pare com ValueError se x, y e dobras tiverem tamanhos diferentes, ou se as dobras não forem
    exatamente 0, 1, ..., k − 1, todas com pelo menos uma linha. Não mude o x que recebeu.
    """
    raise NotImplementedError("escreva erro_nas_dobras")


if __name__ == "__main__":
    sys.exit(rodar(sys.argv[1:], registro=AQUI / "meu-experimento.json", metodo="Lasso", grade=GRADE_LASSO,
                   dividir=dividir_em_dobras, erros=erro_nas_dobras))
