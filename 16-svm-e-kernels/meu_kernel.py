"""Exercício da aula «SVM e kernels»: o alcance do RBF e os pedidos que seguram a reta.

Escreva as duas funções abaixo, sem importar as do fronteiras.py. Depois rode, de dentro da
pasta da aula:

    python3 testar_meu_kernel.py
    python3 meu_kernel.py

Antes de você escrever as funções, os dois param com NotImplementedError. É o esperado.
"""
import math

import numpy as np

from fronteiras import (DEZ, DUAS, MAPA, br, carregar_entregas, carregar_pedidos, contar, cortes, k_linear,
                        pesos, sinal, texto_cortes, treinar)


def kernel_rbf(A, B, gamma):
    """Kernel RBF entre cada linha de A e cada linha de B.

    A: array do NumPy com uma linha por pedido (n linhas); B: outro array com as mesmas colunas
    (m linhas). Retorna um array n × m em que a posição [i, j] é exp(−gamma × d²), com d² a
    soma dos quadrados das diferenças entre a linha i de A e a linha j de B.
    Para com ValueError se gamma não for maior que zero ou se A e B não tiverem o mesmo número de
    colunas. Não muda o que recebeu.
    """
    raise NotImplementedError("escreva kernel_rbf(A, B, gamma)")


def decisao(modelo, X):
    """f(x) do SVM treinado para cada linha de X.

    modelo: o dicionário de treinar(), com X, y (−1 e +1), alfa, b e kernel.
    Retorna um array com um f por linha de X: a soma, nos pedidos do treino, de
    alfa × y × kernel(pedido do treino, x), mais b. Usa o kernel do próprio modelo.
    Para com ValueError se X não tiver o mesmo número de colunas do treino.
    """
    raise NotImplementedError("escreva decisao(modelo, X)")

# ---- daqui para baixo já está pronto ----

GAMMAS = (0.015, 0.1, 1.0, 10.0)
DISTANCIAS = (0.5, 1.0, 3.0, 5.0, 10.0)


def parte_a(marco, abril):
    print("A. O mapa de entregas com o RBF, C = 1 e gamma diferentes")
    X, y = marco[MAPA].to_numpy(float), sinal(marco["cancelado"])
    Xa, ya = abril[MAPA].to_numpy(float), sinal(abril["cancelado"])
    print("  gamma   vetores de suporte   erros em março   erros em abril   onde corta a linha leste-oeste (km)")
    for g in GAMMAS:
        m = treinar(X, y, lambda P, Q, g=g: kernel_rbf(P, Q, g), 1.0)
        em, ea = contar(decisao(m, X), y)[2], contar(decisao(m, Xa), ya)[2]
        pegos, falsos, _ = contar(decisao(m, Xa), ya)
        longe = decisao(m, np.array([[0.0, 30.0]]))[0]
        print(f"  {br(g, 3):>6}   {int((m['alfa'] > 0).sum()):>18}   {em:>14}   {ea:>14}   "
              f"{texto_cortes(cortes(lambda P, m=m: decisao(m, P)))}")
        print(f"          abril: cancelados perdidos {int((ya > 0).sum()) - pegos}, entregues marcados {falsos}; "
              f"b = {br(m['b'], 3)}; f a 30 km do centro, longe de todos = {br(longe, 3)}")


def parte_b():
    print("\nB. Quanto um pedido do treino pesa em outro a cada distância: kernel_rbf")
    print("  gamma  " + "".join(f"{br(d, 1) + ' km':>10}" for d in DISTANCIAS))
    for g in GAMMAS:
        ks = [kernel_rbf(np.array([[0.0, 0.0]]), np.array([[d, 0.0]]), g)[0, 0] for d in DISTANCIAS]
        print(f"  {br(g, 3):>5}  " + "".join(f"{br(k, 3):>10}" for k in ks))


def parte_c(ped):
    print("\nC. A reta dos dez, tirando um pedido de cada vez (C = 1000)")
    for tirar in (None, 36, 1044):
        ids = [p for p in DEZ if p != tirar]
        X, y = ped.loc[ids, DUAS].to_numpy(float), sinal(ped.loc[ids, "cancelado"])
        m = treinar(X, y, k_linear, 1000.0)
        w = pesos(m)
        sv = [ids[k] for k in np.where(m["alfa"] > 0)[0]]
        nome = "os dez" if tirar is None else f"sem o {tirar}"
        linha = (f"  {nome:<10} f = {br(w[0], 3)} × prazo {'−' if w[1] < 0 else '+'} {br(abs(w[1]), 3)} × anteriores "
                 f"{'−' if m['b'] < 0 else '+'} {br(abs(m['b']), 3)}; largura {br(2 / math.hypot(*w))}; "
                 f"vetores de suporte {', '.join(map(str, sv))}")
        if tirar is not None:
            f = decisao(m, ped.loc[[tirar], DUAS].to_numpy(float))[0]
            linha += f"; f do {tirar} nessa reta = {br(f)}"
        print(linha)


def main():
    marco, abril = carregar_entregas()
    parte_a(marco, abril)
    parte_b()
    parte_c(carregar_pedidos())


if __name__ == "__main__":
    main()
