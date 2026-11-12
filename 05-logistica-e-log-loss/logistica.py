"""As contas da aula «Logística e log-loss», com os pedidos da Loja Exemplo.

Rode de dentro da pasta da aula:  python3 logistica.py

Parte 1 a 5: quatro pedidos de janeiro a março e o chute score = -3 + 0,2 × prazo.
Parte 6 e 7: os 1155 pedidos de janeiro a março, os mesmos do treino do curso anterior.
Abril, maio e junho não entram aqui.
A saída é igual toda vez: não tem sorteio.
"""
import numpy as np
import pandas as pd

CAMINHO = "../dados/loja/pedidos.csv"
FIM_TREINO = "2026-03-31"
QUATRO = [10, 41, 9, 25]          # os quatro pedidos das contas à mão
A0, B0 = -3.0, 0.2                # o chute: score = -3 + 0,2 × prazo
TAXA = 0.05
MARCAS = (1, 10, 100, 1000, 10000, 20000)


def br(v, casas=4):
    """Número com vírgula, como na aula. Zero arredondado sai sem sinal."""
    texto = f"{v:.{casas}f}"
    if float(texto) == 0:
        texto = f"{0.0:.{casas}f}"
    return texto.replace(".", ",")


def sigmoide(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def perdas(y, p):
    """A perda de cada pedido: -ln da nota que o modelo deu para o que aconteceu."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(y, p):
    return float(np.mean(perdas(y, p)))


def gradiente(a, b, x, y):
    p = sigmoide(a + b * x)
    return float(np.mean(p - y)), float(np.mean((p - y) * x))


def main():
    base = pd.read_csv(CAMINHO, parse_dates=["data_pedido"])
    treino = base[base["data_pedido"] <= pd.Timestamp(FIM_TREINO)]
    quatro = treino.set_index("id_pedido").loc[QUATRO]
    x = quatro["prazo_prometido_dias"].to_numpy(dtype=float)
    y = quatro["cancelado"].to_numpy(dtype=float)

    print("1. Quatro pedidos de janeiro a março e o chute: score = -3 + 0,2 × prazo")
    print("   pedido  prazo  cancelou   score    nota   perda")
    z = A0 + B0 * x
    p = sigmoide(z)
    for i, pid in enumerate(QUATRO):
        print(f"   {pid:>6}  {int(x[i]):>5}  {'sim' if y[i] else 'não':>8}  {br(z[i], 2):>6}  {br(p[i]):>6}  {br(perdas(y, p)[i]):>6}")
    print(f"   log-loss do chute: {br(log_loss(y, p))}")
    parte = y.mean()
    print(f"   log-loss da nota {br(parte, 2)} para todos (a parte que cancelou): {br(log_loss(y, np.full(4, parte)))}")
    print()

    print("2. A sigmoide: score vira nota entre 0 e 1")
    zs = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    print("   score " + " ".join(f"{br(v, 0):>6}" for v in zs))
    print("   nota  " + " ".join(f"{br(v, 3):>6}" for v in sigmoide(zs)))
    print(f"   e de volta: ln(0,5 ÷ 0,5) = {br(np.log(0.5 / 0.5), 2)}; ln(0,2689 ÷ 0,7311) = {br(np.log(0.2689 / 0.7311), 2)}")
    print()

    print("3. Quanto custa cada nota, pela log-loss e pelo erro ao quadrado")
    print("   nota dada ao que aconteceu   log-loss   erro ao quadrado")
    for q in (0.99, 0.9, 0.5, 0.2, 0.1, 0.01):
        print(f"   {br(q, 2):>26}   {br(-np.log(q)):>8}   {br((1 - q) ** 2):>16}")
    print()

    print("4. O gradiente da log-loss nos quatro pedidos")
    d_a, d_b = gradiente(A0, B0, x, y)
    print("   nota - cancelou:           " + "  ".join(br(v) for v in p - y))
    print("   (nota - cancelou) × prazo: " + "  ".join(br(v) for v in (p - y) * x))
    print(f"   derivada no -3: {br(d_a)} | derivada no 0,2: {br(d_b)}")
    h = 0.0001
    num_a = (log_loss(y, sigmoide(A0 + h + B0 * x)) - log_loss(y, sigmoide(A0 - h + B0 * x))) / (2 * h)
    num_b = (log_loss(y, sigmoide(A0 + (B0 + h) * x)) - log_loss(y, sigmoide(A0 + (B0 - h) * x))) / (2 * h)
    print(f"   mexendo 0,0001 para cima e para baixo: no -3 mudou ÷ mexi = {br(num_a)}; no 0,2 = {br(num_b)}")
    print()

    print(f"5. Um passo com taxa {br(TAXA, 2)}")
    a1, b1 = A0 - TAXA * d_a, B0 - TAXA * d_b
    print(f"   a: -3 - {br(TAXA, 2)} × {br(d_a)} = {br(a1)}")
    print(f"   b: 0,2 - {br(TAXA, 2)} × {br(d_b)} = {br(b1)}")
    p1 = sigmoide(a1 + b1 * x)
    print("   notas novas: " + "  ".join(br(v) for v in p1))
    print(f"   log-loss: {br(log_loss(y, p))} antes, {br(log_loss(y, p1))} depois")
    print()

    xt = treino["prazo_prometido_dias"].to_numpy(dtype=float)
    yt = treino["cancelado"].to_numpy(dtype=float)
    parte_t = yt.mean()
    print(f"6. Os {len(yt)} pedidos de janeiro a março: {int(yt.sum())} cancelaram ({br(100 * parte_t, 1)}%)")
    print(f"   log-loss do chute -3 + 0,2 × prazo: {br(log_loss(yt, sigmoide(A0 + B0 * xt)))}")
    print(f"   log-loss da nota {br(parte_t, 4)} para todos: {br(log_loss(yt, np.full(len(yt), parte_t)))}")
    d_a, d_b = gradiente(A0, B0, xt, yt)
    print(f"   gradiente no chute: ({br(d_a)}; {br(d_b)})")
    print(f"   passos com taxa {br(TAXA, 2)}, saindo do chute:")
    print("      passo   log-loss        a        b")
    a, b = A0, B0
    for k in range(1, MARCAS[-1] + 1):
        d_a, d_b = gradiente(a, b, xt, yt)
        a, b = a - TAXA * d_a, b - TAXA * d_b
        if k in MARCAS:
            print(f"   {k:>8}   {br(log_loss(yt, sigmoide(a + b * xt)), 5):>8}  {br(a):>7}  {br(b):>7}")
    d_a, d_b = gradiente(a, b, xt, yt)
    print(f"   gradiente no fim: ({br(d_a, 6)}; {br(d_b, 6)})")
    print()

    print(f"7. O que as notas do fim dizem: score = {br(a, 2)} + {br(b, 3)} × prazo")
    pt = sigmoide(a + b * xt)
    print(f"   soma das notas dos {len(yt)} pedidos: {br(pt.sum(), 1)} | pedidos que cancelaram: {int(yt.sum())}")
    print("   prazo  pedidos  cancelaram  nota do modelo")
    for prazo in (2, 8, 15):
        g = xt == prazo
        print(f"   {prazo:>5}  {int(g.sum()):>7}  {br(100 * yt[g].mean(), 1):>9}%  {br(100 * sigmoide(a + b * prazo), 1):>13}%")


if __name__ == "__main__":
    main()
