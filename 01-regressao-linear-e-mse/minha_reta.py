"""Exercício da aula «Regressão linear e MSE»: o tempo de separação dos pedidos da Bia.

Escreva as duas funções marcadas com NotImplementedError, sem loop em Python dentro delas.
Depois rode, de dentro da pasta da aula:

    python3 testar_minha_reta.py
    python3 minha_reta.py

Enquanto as funções não existem, os dois comandos param com NotImplementedError.
"""
import numpy as np

# Os cinco pedidos que a Bia cronometrou: itens e minutos de separação.
ITENS = [1, 2, 4, 5, 8]
MINUTOS = [5, 9, 13, 13, 20]


def mse(a, b, x, y):
    """Retorna o MSE da reta a + b * x nos pontos (x, y).

    x e y podem chegar como lista. Pare com ValueError se tiverem tamanhos diferentes.
    """
    raise NotImplementedError("escreva mse(a, b, x, y)")


def melhor_reta(x, y):
    """Retorna (a, b) da reta de menor MSE, pela conta das médias da aula.

    Pare com ValueError se x e y tiverem tamanhos diferentes ou se os x forem todos iguais.
    """
    raise NotImplementedError("escreva melhor_reta(x, y)")


def br(numero, casas=2):
    return f"{numero:.{casas}f}".replace(".", ",")


if __name__ == "__main__":
    x = np.asarray(ITENS, dtype=float)
    y = np.asarray(MINUTOS, dtype=float)
    for nome, a, b in [("regra da Bia", 3, 2.5), ("regra do Pedro", 12, 0)]:
        print(f"{nome}: minutos = {br(a)} + {br(b)} × itens | MSE {br(mse(a, b, x, y), 4)}")
    a, b = melhor_reta(x, y)
    residuo = y - (a + b * x)
    print(f"melhor reta: minutos = {br(a)} + {br(b)} × itens | MSE {br(mse(a, b, x, y), 4)}")
    print("  resíduos " + "  ".join(br(r) for r in residuo))
    print(f"  soma dos resíduos {br(residuo.sum())} | soma de resíduo × itens {br((residuo * x).sum())}")
