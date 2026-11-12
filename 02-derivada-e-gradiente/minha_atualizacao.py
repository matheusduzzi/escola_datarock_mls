"""Exercício da aula «Derivada e gradiente»: um passo do gradiente nos pedidos da Rê.

Escreva as duas funções marcadas com NotImplementedError, sem loop em Python dentro delas.
Depois rode, de dentro da pasta da aula:

    python3 testar_minha_atualizacao.py
    python3 minha_atualizacao.py

Enquanto as funções não existem, os dois comandos param com NotImplementedError.
"""
import numpy as np

# Os cinco pedidos de Papelaria que a Rê mandou: itens e valor pago, em reais.
ITENS = [1, 2, 3, 4, 5]
VALOR = [28, 35, 52, 55, 70]


def erro_quadratico_medio(y, previsto):
    """A média de (y - previsto) ao quadrado. É a mesma função da tarefa 2 do diagnóstico."""
    y = np.asarray(y, dtype=float)
    previsto = np.asarray(previsto, dtype=float)
    if y.shape != previsto.shape:
        raise ValueError(f"tamanhos diferentes: {y.shape} e {previsto.shape}")
    return float(np.mean((y - previsto) ** 2))


def gradiente(a, b, x, y):
    """Retorna (derivada do MSE no a, derivada do MSE no b) para a reta a + b * x.

    x e y podem chegar como lista. Pare com ValueError se tiverem tamanhos diferentes.
    """
    raise NotImplementedError("escreva gradiente(a, b, x, y)")


def passo(a, b, x, y, taxa):
    """Retorna (a novo, b novo) depois de um passo do gradiente com essa taxa.

    Pare com ValueError se a taxa não for maior que zero.
    """
    raise NotImplementedError("escreva passo(a, b, x, y, taxa)")


if __name__ == "__main__":
    a, b = 20.0, 10.0
    x = np.asarray(ITENS, dtype=float)
    print(f"Reta de partida: valor = {a:g} + {b:g} × itens")
    print(f"  MSE {erro_quadratico_medio(VALOR, a + b * x):.4f}")
    d_a, d_b = gradiente(a, b, ITENS, VALOR)
    print(f"  gradiente ({d_a:.4f}; {d_b:.4f})")
    a1, b1 = passo(a, b, ITENS, VALOR, 0.01)
    print(f"Um passo com taxa 0.01: a = {a1:.4f}, b = {b1:.4f}")
    print(f"  MSE {erro_quadratico_medio(VALOR, a1 + b1 * x):.4f}")
    print()
    print("Trinta passos, MSE depois de cada um dos 3 primeiros e do último:")
    for taxa in (0.01, 0.05, 0.1):
        a_t, b_t = a, b
        historico = []
        for _ in range(30):
            a_t, b_t = passo(a_t, b_t, ITENS, VALOR, taxa)
            historico.append(erro_quadratico_medio(VALOR, a_t + b_t * x))
        primeiros = "  ".join(f"{v:.2f}" for v in historico[:3])
        print(f"  taxa {taxa:<4}  {primeiros}  ...  {historico[-1]:.2f}")
