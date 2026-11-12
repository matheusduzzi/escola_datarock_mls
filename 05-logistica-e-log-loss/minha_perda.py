"""Exercício da aula «Logística e log-loss»: a regra do boleto que a Bia recebeu.

Escreva as três funções marcadas com NotImplementedError, sem loop em Python dentro delas.
Depois rode, de dentro da pasta da aula:

    python3 testar_minha_perda.py
    python3 minha_perda.py

Enquanto as funções não existem, os dois comandos param com NotImplementedError.
"""
import numpy as np
import pandas as pd

CAMINHO = "../dados/loja/pedidos.csv"
FIM_TREINO = "2026-03-31"

# A regra que chegou para a Bia: score = -2 + 1 × boleto (boleto vale 1, pix e cartão valem 0).
A0, B0 = -2.0, 1.0
TAXA = 1.0
PASSOS = 500

# Os seis pedidos de janeiro a março que a Bia separou, como estão em pedidos.csv.
PEDIDOS = [15, 57, 12, 14, 18, 26]
BOLETO = [1, 1, 0, 0, 0, 0]
CANCELOU = [1, 0, 1, 0, 0, 0]


def sigmoide(z):
    """Retorna a nota entre 0 e 1 de cada score: 1 / (1 + e elevado a -score).

    z pode chegar como número, lista ou array.
    """
    raise NotImplementedError("escreva sigmoide(z)")


def log_loss(y, p):
    """Retorna a log-loss média: a média de -ln da nota dada ao que aconteceu.

    y tem 1 para cancelado e 0 para entregue; p tem as notas. Os dois podem chegar como lista.
    Pare com ValueError se os tamanhos forem diferentes, se y tiver valor que não é 0 nem 1,
    ou se alguma nota estiver fora de 0 a 1. Corte as notas em 1e-15 e 1 - 1e-15 antes do ln,
    para que uma nota 0 ou 1 dê um número grande, e não infinito.
    """
    raise NotImplementedError("escreva log_loss(y, p)")


def gradiente(a, b, x, y):
    """Retorna (derivada da log-loss no a, derivada no b) para o score a + b * x.

    Pare com ValueError se x e y tiverem tamanhos diferentes.
    """
    raise NotImplementedError("escreva gradiente(a, b, x, y)")


def br(v, casas=4):
    texto = f"{v:.{casas}f}"
    if float(texto) == 0:
        texto = f"{0.0:.{casas}f}"
    return texto.replace(".", ",")


if __name__ == "__main__":
    x = np.asarray(BOLETO, dtype=float)
    y = np.asarray(CANCELOU, dtype=float)
    print("Seis pedidos e a regra score = -2 + 1 × boleto")
    p = sigmoide(A0 + B0 * x)
    for pid, xi, yi, pi in zip(PEDIDOS, BOLETO, CANCELOU, p):
        print(f"  pedido {pid:>3}  boleto {xi}  cancelou {yi}  nota {br(pi)}")
    print(f"  log-loss da regra: {br(log_loss(y, p))}")
    print(f"  log-loss da nota 0,2 para todos: {br(log_loss(y, np.full(len(y), 0.2)))}")
    d_a, d_b = gradiente(A0, B0, x, y)
    print(f"  gradiente ({br(d_a)}; {br(d_b)})")
    a1, b1 = A0 - TAXA * d_a, B0 - TAXA * d_b
    print(f"  um passo com taxa {br(TAXA, 0)}: a = {br(a1)}, b = {br(b1)}")
    print(f"  log-loss depois do passo: {br(log_loss(y, sigmoide(a1 + b1 * x)))}")
    print()

    base = pd.read_csv(CAMINHO, parse_dates=["data_pedido"])
    treino = base[base["data_pedido"] <= pd.Timestamp(FIM_TREINO)]
    xt = (treino["forma_pagamento"] == "boleto").to_numpy(dtype=float)
    yt = treino["cancelado"].to_numpy(dtype=float)
    print(f"Os {len(yt)} pedidos de janeiro a março")
    print(f"  log-loss da regra: {br(log_loss(yt, sigmoide(A0 + B0 * xt)))}")
    print(f"  log-loss da nota 0,2 para todos: {br(log_loss(yt, np.full(len(yt), 0.2)))}")
    d_a, d_b = gradiente(A0, B0, xt, yt)
    print(f"  gradiente na regra ({br(d_a)}; {br(d_b)})")
    a, b = A0, B0
    for _ in range(PASSOS):
        d_a, d_b = gradiente(a, b, xt, yt)
        a, b = a - TAXA * d_a, b - TAXA * d_b
    print(f"  depois de {PASSOS} passos com taxa {br(TAXA, 0)}: score = {br(a, 3)} + {br(b, 3)} × boleto")
    print(f"  log-loss: {br(log_loss(yt, sigmoide(a + b * xt)))}")
    for nome, valor in (("pix ou cartão", 0.0), ("boleto", 1.0)):
        g = xt == valor
        print(f"  {nome:>13}: {int(g.sum())} pedidos, {br(100 * yt[g].mean(), 1)}% cancelaram, "
              f"nota do modelo {br(100 * float(sigmoide(a + b * valor)), 1)}%")
