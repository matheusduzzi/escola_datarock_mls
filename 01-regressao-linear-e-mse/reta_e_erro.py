"""As contas da aula «Regressão linear e MSE» com os quatro pedidos.

Rode de dentro da pasta da aula:  python3 reta_e_erro.py
A saída é a mesma toda vez: só faz contas com os pedidos abaixo, sem sorteio.
"""
import numpy as np

# Os quatro pedidos: quantos itens e quanto o cliente pagou, em reais.
ITENS = np.array([1, 2, 4, 5], dtype=float)
VALOR = np.array([70, 85, 150, 160], dtype=float)
NOMES = ["A", "B", "C", "D"]


def br(numero, casas=2):
    """Número com vírgula decimal e ponto no milhar, como na aula."""
    texto = f"{numero:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def prever(a, b, x):
    return a + b * x


def mse(a, b, x, y):
    return float(np.mean((y - prever(a, b, x)) ** 2))


def linha(rotulo, valores, casas=2):
    print(f"  {rotulo:<16}" + "".join(f"{br(v, casas):>10}" for v in valores))


def mostrar_reta(a, b, titulo):
    previsto = prever(a, b, ITENS)
    residuo = VALOR - previsto
    print(f"{titulo}: valor = {br(a)} + {br(b)} × itens")
    linha("previsto", previsto)
    linha("resíduo", residuo)
    linha("quadrado", residuo ** 2)
    print(f"  soma dos resíduos {br(residuo.sum())} | soma dos quadrados {br((residuo ** 2).sum())}"
          f" | MSE {br(mse(a, b, ITENS, VALOR), 4)}")


print("Quatro pedidos: " + ", ".join(f"{n} {int(i)} itens R$ {int(v)}" for n, i, v in zip(NOMES, ITENS, VALOR)))
print(" " * 18 + "".join(f"{'pedido ' + n:>10}" for n in NOMES))
print()

print("1. Duas retas e o erro de cada uma")
mostrar_reta(40, 25, "  reta 1")
mostrar_reta(30, 27, "  reta 2")
print()

print("2. Uma reta que ignora os itens e prevê a média para todo pedido")
mostrar_reta(VALOR.mean(), 0, "  reta plana")
print()

print("3. O MSE de várias retas: intercepto na linha, inclinação na coluna")
inclinacoes = [23, 24, 25, 26, 27]
print("  " + " " * 16 + "".join(f"{'b = ' + str(b):>10}" for b in inclinacoes))
for a in [35, 40, 45, 50]:
    linha(f"a = {a}", [mse(a, b, ITENS, VALOR) for b in inclinacoes])
print()

print("4. Mexer em um número de cada vez, saindo da reta 1")
mostrar_reta(50, 25, "  a sobe 10")
mostrar_reta(40, 26, "  b sobe 1")
print()

print("5. A reta de menor MSE, pela conta das médias")
mx, my = ITENS.mean(), VALOR.mean()
dx, dy = ITENS - mx, VALOR - my
print(f"  média dos itens {br(mx)} | média do valor {br(my)}")
linha("itens - média", dx)
linha("valor - média", dy)
linha("produto", dx * dy)
linha("(itens - média)²", dx ** 2)
b = (dx * dy).sum() / (dx ** 2).sum()
a = my - b * mx
print(f"  b = {br((dx * dy).sum())} ÷ {br((dx ** 2).sum())} = {br(b)}")
print(f"  a = {br(my)} - {br(b)} × {br(mx)} = {br(a)}")
mostrar_reta(a, b, "  melhor reta")
residuo = VALOR - prever(a, b, ITENS)
print(f"  soma de resíduo × itens {br((residuo * ITENS).sum())}")
print()

print("6. O NumPy acha a mesma reta")
coef = np.polyfit(ITENS, VALOR, 1)
print(f"  np.polyfit(itens, valor, 1) = [{br(coef[0])}; {br(coef[1])}]  (inclinação primeiro, intercepto depois)")
