"""Ridge, Lasso e Elastic Net nos pedidos da Loja Exemplo, só com o pandas e o NumPy.

Rode de dentro da pasta da aula:  python3 penalizacao.py

Também serve de módulo para o exercício:

    from penalizacao import carregar, primeiros, ajustar, mse, COLUNAS

O custo que cada ajuste minimiza, com as colunas já padronizadas:

    MSE + l1 × soma(|coeficiente|) + l2 × soma(coeficiente²)

- l1 = 0 e l2 = 0: regressão sem penalidade;
- só l2: Ridge;
- só l1: Lasso;
- os dois: Elastic Net.

O intercepto fica fora da penalidade. Não tem sorteio: a saída sai igual toda vez.
"""
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
BASE_PADRAO = AQUI.parent / "dados" / "valor" / "pedidos_valor.csv"
COLUNAS = ["itens", "peso_kg", "cliente_novo", "cupom", "visitas_antes",
           "prazo_prometido_dias", "hora_pedido", "dia_semana"]


def carregar(caminho=BASE_PADRAO):
    """Lê os pedidos, tira os 12 de empresa (40 itens ou mais) e ordena por data e número."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com dados/ ao lado.")
    dados = pd.read_csv(caminho)
    faltam = [c for c in COLUNAS + ["id_pedido", "data_pedido", "valor"] if c not in dados.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    if dados[COLUNAS + ["valor"]].isna().any().any():
        raise ValueError("há valores vazios nas colunas do modelo")
    varejo = dados[dados["itens"] < 40]
    return varejo.sort_values(["data_pedido", "id_pedido"]).reset_index(drop=True)


def primeiros(dados, mes, n=40):
    """Os n primeiros pedidos do mês (2026) e todos os outros pedidos do semestre."""
    do_mes = dados[dados["data_pedido"].str.startswith(f"2026-{mes:02d}-")]
    if len(do_mes) < n:
        raise ValueError(f"o mês {mes} tem {len(do_mes)} pedidos, menos que {n}")
    treino = do_mes.head(n)
    return treino, dados.drop(treino.index)


def padronizar(treino, outros):
    """Tira a média e divide pelo desvio padrão, os dois calculados só no treino."""
    media = treino.mean()
    desvio = treino.std(ddof=0)
    if (desvio == 0).any():
        raise ValueError(f"coluna sem variação no treino: {', '.join(desvio.index[desvio == 0])}")
    return (treino - media) / desvio, (outros - media) / desvio, media, desvio


def ajustar(z, y, l1=0.0, l2=0.0):
    """Coeficientes nas colunas padronizadas e o intercepto. z: DataFrame ou array; y: valores."""
    if l1 < 0 or l2 < 0:
        raise ValueError("as penalidades não podem ser negativas")
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = z.shape
    intercepto = y.mean()
    yc = y - intercepto
    if l1 == 0:
        # sem a parte do módulo, a conta é direta: (zᵀz/n + l2 × I) b = zᵀy/n
        a = z.T.dot(z) / n + l2 * np.eye(p)
        if np.linalg.matrix_rank(a) < p:
            raise ValueError("colunas repetidas e sem penalidade: o ajuste não tem um só resultado")
        return np.linalg.solve(a, z.T.dot(yc) / n), intercepto
    # com o módulo, uma coluna de cada vez até nada mudar (descida por coordenada)
    var = (z ** 2).mean(axis=0)
    b = np.zeros(p)
    for _ in range(20000):
        maior = 0.0
        for j in range(p):
            r = yc - z.dot(b) + z[:, j] * b[j]
            c = np.mean(z[:, j] * r)
            novo = np.sign(c) * max(abs(c) - l1 / 2, 0.0) / (var[j] + l2)
            maior = max(maior, abs(novo - b[j]))
            b[j] = novo
        if maior < 1e-10:
            return b, intercepto
    raise RuntimeError("a descida por coordenada não parou em 20000 voltas")


def mse(z, y, coef, intercepto):
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((y - intercepto - z.dot(coef)) ** 2))


def br(x, casas=2):
    """Número com vírgula decimal e ponto de milhar, como na aula."""
    if round(x, casas) == 0:
        x = 0.0
    s = f"{x:,.{casas}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def tabela(titulo, linhas, z, y, zo, yo, referencia=None):
    """Uma coluna por ajuste: coeficientes padronizados, MSE nos 40 e nos outros pedidos."""
    print(titulo)
    extra = f"{'todos':>11}" if referencia is not None else ""
    print("  " + f"{'coluna':22}" + "".join(f"{nome:>11}" for nome, _, _ in linhas) + extra)
    ajustes = [ajustar(z, y, l1, l2) for _, l1, l2 in linhas]
    for j, col in enumerate(COLUNAS):
        extra = f"{br(referencia[j], 1):>11}" if referencia is not None else ""
        print("  " + f"{col:22}" + "".join(f"{br(b[j], 1):>11}" for b, _ in ajustes) + extra)
    print("  " + f"{'MSE nos 40':22}" + "".join(f"{br(mse(z, y, b, a), 0):>11}" for b, a in ajustes))
    print("  " + f"{'MSE nos outros':22}" + "".join(f"{br(mse(zo, yo, b, a), 0):>11}" for b, a in ajustes))
    print()
    return ajustes


def uma_coluna():
    """Seção 3 da aula: a conta à mão com cinco pedidos, com o peso em kg e em gramas."""
    peso = np.array([1, 2, 3, 4, 5], dtype=float)
    valor = np.array([60, 90, 100, 150, 150], dtype=float)
    print("2. Uma coluna, à mão: cinco pedidos")
    print("  peso (kg)  1  2  3  4  5 | valor (R$)  60  90  100  150  150")
    contas = []
    for x in (peso, peso * 1000):
        xc, yc = x - x.mean(), valor - valor.mean()
        contas.append((np.mean(xc ** 2), np.mean(xc * yc)))
    print(f"  {'':16}{'peso em kg':>14}{'peso em gramas':>30}")
    print(f"  {'variância':16}{br(contas[0][0], 0):>14}{br(contas[1][0], 0):>30}")
    print(f"  {'covariância':16}{br(contas[0][1], 0):>14}{br(contas[1][1], 0):>30}")
    for nome, l1, l2 in (("sem penalidade", 0, 0), ("Ridge alfa 1", 0, 1), ("Ridge alfa 3", 0, 3),
                         ("Lasso alfa 20", 20, 0), ("Lasso alfa 96", 96, 0)):
        b = [np.sign(c) * max(abs(c) - l1 / 2, 0) / (v + l2) for v, c in contas]
        print(f"  {nome:16}{br(b[0]):>8} por kg{br(b[1], 6):>16} por g ={br(b[1] * 1000):>6} por kg")
    desvio = peso.std()
    z = ((peso - peso.mean()) / desvio).reshape(-1, 1)
    print(f"  padronizado (desvio {br(desvio, 4)} kg, ou {br(desvio * 1000, 1)} g: dá igual)")
    for nome, l1, l2 in (("Ridge alfa 1", 0, 1), ("Lasso alfa 20", 20, 0)):
        b, _ = ajustar(z, valor, l1, l2)
        print(f"  {nome:16}{br(b[0]):>8} por desvio ={br(b[0] / desvio):>6} por kg")
    dobro = np.column_stack([z[:, 0], z[:, 0]])
    print("  a mesma coluna duas vezes, padronizada: coeficiente de cada uma")
    for nome, l1, l2 in (("Ridge alfa 1", 0, 1), ("Lasso alfa 20", 20, 0), ("Elastic Net 20 e 1", 20, 1)):
        b, _ = ajustar(dobro, valor, l1, l2)
        print(f"  {nome:20}{br(b[0]):>7} e {br(b[1]):>6}")
    print()


def main():
    dados = carregar()
    treino, outros = primeiros(dados, 1)
    print(f"{len(dados)} pedidos de varejo, de {dados['data_pedido'].min()} a {dados['data_pedido'].max()}")
    print(f"os 40 primeiros de janeiro vão de {treino['data_pedido'].min()} a {treino['data_pedido'].max()}; "
          f"os outros {len(outros)} servem para ver o efeito")
    print(f"correlação entre itens e peso_kg: {br(treino['itens'].corr(treino['peso_kg']), 2)} nos 40, "
          f"{br(dados['itens'].corr(dados['peso_kg']), 2)} nos {len(dados)}")
    print()

    # 1. Sem penalidade, em R$ por unidade de cada coluna
    def em_reais(parte):
        a = np.column_stack([np.ones(len(parte)), parte[COLUNAS].to_numpy(float)])
        return np.linalg.lstsq(a, parte["valor"].to_numpy(float), rcond=None)[0]
    b40, btodos = em_reais(treino), em_reais(dados)
    print("1. Regressão sem penalidade, em R$ por unidade de cada coluna")
    print(f"  {'coluna':22}{'40 de janeiro':>15}{'todos os ' + str(len(dados)):>16}")
    for j, col in enumerate(["intercepto"] + COLUNAS):
        print(f"  {col:22}{br(b40[j]):>15}{br(btodos[j]):>16}")
    print()

    uma_coluna()

    z, zo, media, desvio = padronizar(treino[COLUNAS], outros[COLUNAS])
    y, yo = treino["valor"].to_numpy(float), outros["valor"].to_numpy(float)
    print("3. Média e desvio padrão de cada coluna nos 40 pedidos")
    for col in COLUNAS:
        print(f"  {col:22}{br(media[col]):>9}{br(desvio[col]):>9}")
    print(f"  valor médio (o intercepto de todos os ajustes padronizados): {br(y.mean())}")
    print()

    sem, _ = ajustar(z, y)
    print("  sem penalidade, padronizado e de volta para R$ por unidade: dá o mesmo da parte 1")
    print("  " + "  ".join(f"{c} {br(sem[j] / desvio[c])}" for j, c in enumerate(COLUNAS[:2])))
    print()

    tabela("4. Ridge: coeficientes padronizados (R$ por desvio padrão da coluna)",
           [("sem", 0, 0), ("0,02", 0, 0.02), ("0,1", 0, 0.1), ("0,5", 0, 0.5), ("2", 0, 2)], z, y, zo, yo)
    tabela("5. Lasso: coeficientes padronizados",
           [("sem", 0, 0), ("5", 5, 0), ("10", 10, 0), ("20", 20, 0), ("40", 40, 0)], z, y, zo, yo)
    referencia = [btodos[j + 1] * desvio[c] for j, c in enumerate(COLUNAS)]
    tabela("6. Os três lado a lado, e a regressão sem penalidade em todos os pedidos (na mesma escala)",
           [("sem", 0, 0), ("Ridge", 0, 0.1), ("Lasso", 20, 0), ("Elastic", 15, 0.2)], z, y, zo, yo,
           referencia=referencia)
    print("  Ridge alfa 0,1 | Lasso alfa 20 | Elastic Net l1 15 e l2 0,2 | todos: coeficiente em R$ por")
    print("  unidade nos " + str(len(dados)) + " pedidos, vezes o desvio da coluna nos 40")
    print()

    # 7. Sem padronizar, a unidade muda o resultado
    print("7. Ridge alfa 0,1 sem padronizar, com o peso em três unidades")
    for unidade, fator in (("kg", 1), ("gramas", 1000), ("toneladas", 0.001)):
        bruto = treino[COLUNAS].astype(float).copy()
        bruto["peso_kg"] = bruto["peso_kg"] * fator
        b, _ = ajustar(bruto - bruto.mean(), y, 0, 0.1)
        print(f"  peso em {unidade:10} itens {br(b[0]):>7} por item | peso {br(b[1] * fator):>7} por kg")
    b, _ = ajustar(z, y, 0, 0.1)
    print(f"  {'padronizado':19}itens {br(b[0] / desvio['itens']):>7} por item | peso "
          f"{br(b[1] / desvio['peso_kg']):>7} por kg (igual em qualquer unidade)")
    print()


if __name__ == "__main__":
    main()
