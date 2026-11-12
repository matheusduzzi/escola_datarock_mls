"""As contas da aula «Derivada e gradiente»: quatro pedidos, uma reta e o passo que diminui o erro.

Rode de dentro da pasta da aula:  python3 gradiente.py
Só usa o NumPy. A saída sai igual toda vez: não tem sorteio.
"""
import numpy as np

# Os quatro pedidos da aula 01: quantos itens e quanto o cliente pagou, em reais.
itens = np.array([1, 2, 4, 5], dtype=float)
valor = np.array([70, 85, 150, 160], dtype=float)


def br(v, casas=2):
    """Número com vírgula decimal e ponto de milhar, como no texto da aula."""
    texto = f"{v:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def mse(a, b, x=itens, y=valor):
    """Média dos quadrados dos resíduos da reta a + b * x."""
    residuo = y - (a + b * x)
    return np.mean(residuo ** 2)


def gradiente(a, b, x=itens, y=valor):
    """Quanto o MSE muda por unidade de a e por unidade de b, perto de (a, b)."""
    residuo = y - (a + b * x)
    d_a = -2 * np.mean(residuo)
    d_b = -2 * np.mean(residuo * x)
    return d_a, d_b


def linha(nome, valores, casas=2):
    return f"  {nome:<10}" + "".join(f"{br(v, casas):>9}" for v in valores)


a0, b0 = 40.0, 25.0
print("Quatro pedidos: itens 1, 2, 4, 5 | valor pago 70, 85, 150, 160")
print()
print("1. A reta de partida: valor = 40 + 25 × itens")
previsto = a0 + b0 * itens
print(linha("previsto", previsto))
print(linha("resíduo", valor - previsto))
print(f"  MSE {br(mse(a0, b0))}")
print()

print("2. Mexer um pouco na inclinação, com o 40 parado")
print("     b novo        MSE    mudou   mudou ÷ quanto mexi")
for h in (0.1, 0.01, 0.001):
    novo = mse(a0, b0 + h)
    print(f"  {br(b0 + h, 3):>9} {br(novo, 6):>11} {br(novo - mse(a0, b0), 6):>9} {br((novo - mse(a0, b0)) / h, 4):>12}")
h = 0.001
print(f"  E no 40, com o 25 parado: 40,001 dá MSE {br(mse(a0 + h, b0), 6)}, "
      f"mudou ÷ 0,001 = {br((mse(a0 + h, b0) - mse(a0, b0)) / h, 4)}")
print()

print("3. A conta direta, sem mexer em nada")
residuo = valor - (a0 + b0 * itens)
d_a, d_b = gradiente(a0, b0)
print(f"  soma dos resíduos {br(residuo.sum())} | soma de resíduo × itens {br((residuo * itens).sum())}")
print(f"  derivada no 40: -2 × {br(residuo.sum())} ÷ 4 = {br(d_a)}")
print(f"  derivada no 25: -2 × {br((residuo * itens).sum())} ÷ 4 = {br(d_b)}")
print(f"  gradiente ({br(d_a)}; {br(d_b)})")
print()

print("4. Um passo com taxa 0,01")
taxa = 0.01
a1, b1 = a0 - taxa * d_a, b0 - taxa * d_b
print(f"  a: 40 - 0,01 × ({br(d_a)}) = {br(a1, 3)}")
print(f"  b: 25 - 0,01 × ({br(d_b)}) = {br(b1, 3)}")
print(f"  MSE: {br(mse(a0, b0))} antes, {br(mse(a1, b1), 4)} depois")
print()

print("5. Trinta passos com três taxas, saindo de 40 e 25")
print("     taxa   passo 1   passo 2   passo 3            passo 30          a          b")
for taxa in (0.01, 0.05, 0.1):
    a, b = a0, b0
    historico = []
    for _ in range(30):
        d_a, d_b = gradiente(a, b)
        a, b = a - taxa * d_a, b - taxa * d_b
        historico.append(mse(a, b))
    print(f"  {br(taxa):>7}" + "".join(f"{br(v):>10}" for v in historico[:3])
          + f"{br(historico[-1]):>20} {br(a):>10} {br(b):>10}")
print()

print("6. A reta de menor MSE para esses quatro pedidos")
b_min = np.sum((itens - itens.mean()) * (valor - valor.mean())) / np.sum((itens - itens.mean()) ** 2)
a_min = valor.mean() - b_min * itens.mean()
print(f"  valor = {br(a_min)} + {br(b_min)} × itens | MSE {br(mse(a_min, b_min), 4)}"
      f" | gradiente ({br(gradiente(a_min, b_min)[0] + 0.0)}; {br(gradiente(a_min, b_min)[1] + 0.0)})")
print()

print("7. Os mesmos pedidos com os itens contados a partir da média (itens - 3)")
centrado = itens - itens.mean()
print("     taxa   passo 1   passo 2   passo 3            passo 30")
for taxa in (0.05, 0.1):
    a, b = a0 + b0 * itens.mean(), b0  # a mesma reta de partida, escrita com os itens centrados
    historico = []
    for _ in range(30):
        d_a, d_b = gradiente(a, b, centrado)
        a, b = a - taxa * d_a, b - taxa * d_b
        historico.append(mse(a, b, centrado))
    print(f"  {br(taxa):>7}" + "".join(f"{br(v):>10}" for v in historico[:3]) + f"{br(historico[-1]):>20}")
