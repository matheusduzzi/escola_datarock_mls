"""As contas da aula «Valor do pedido», nos 1500 pedidos entregues da Loja Exemplo.

    python3 valor_pedido.py

Lê ../dados/valor/pedidos_valor.csv e mostra, em ordem: as médias com e sem cupom, a regressão
só com o cupom, com o cupom e o tipo de cliente, o modelo da loja, os resíduos, o modelo sem os
pedidos de empresa, o modelo com os itens por categoria, o caminho do coeficiente do cupom e o
intervalo dos coeficientes. Sai igual toda vez.
"""
import sys

from regressao import ajustar, br, carregar, intervalo, residuos_por

MODELO_LOJA = ["itens", "categoria", "cliente_novo", "cupom", "canal"]
MODELO_FINAL = MODELO_LOJA + ["itens_por_categoria"]


def coeficientes(ajuste, nomes=None):
    for nome, valor in ajuste.coeficientes.items():
        if nomes is None or nome in nomes:
            print(f"  {nome:<20}{br(valor):>10}")


def tabela(t):
    for chave, linha in t.iterrows():
        print(f"  {str(chave):<12}{int(linha['pedidos']):>8}{br(linha['resíduo médio']):>12}")


def cruzada(ajuste, dados):
    from regressao import faixa_de_itens
    t = ajuste.residuo.groupby([faixa_de_itens(dados["itens"]), dados["categoria"]], observed=True).mean()
    t = t.unstack()
    print(f"  {'itens':<12}" + "".join(f"{c:>11}" for c in t.columns))
    for faixa, linha in t.iterrows():
        print(f"  {str(faixa):<12}" + "".join(f"{('' if v != v else br(v, 0)):>11}" for v in linha))


def main():
    dados = carregar()
    print(f"{len(dados)} pedidos entregues, de {dados['data_pedido'].min()} a {dados['data_pedido'].max()}")

    print("\n1. Valor médio com e sem cupom")
    for tipo, nome in [(None, "todos"), (0, "recorrentes"), (1, "novos")]:
        parte = dados if tipo is None else dados[dados["cliente_novo"] == tipo]
        com, sem = parte[parte["cupom"] == 1], parte[parte["cupom"] == 0]
        print(f"  {nome:<12} com cupom {br(com['valor'].mean()):>7} ({len(com)})"
              f"   sem cupom {br(sem['valor'].mean()):>7} ({len(sem)})")
    novos_com_cupom = dados.loc[dados["cupom"] == 1, "cliente_novo"].mean()
    print(f"  pedidos com cupom que são de cliente novo: {br(100 * novos_com_cupom, 1)}%")

    print("\n2. Regressão só com o cupom")
    coeficientes(ajustar(dados, ["cupom"]))

    print("\n3. Cupom e tipo de cliente")
    coeficientes(ajustar(dados, ["cupom", "cliente_novo"]))

    print("\n4. O modelo da loja, nos 1500 pedidos")
    loja = ajustar(dados, MODELO_LOJA)
    coeficientes(loja)

    print("\n5. Resíduo médio por faixa de itens (valor pago menos previsto)")
    print(f"  {'itens':<12}{'pedidos':>8}{'resíduo':>12}")
    tabela(residuos_por(loja, dados, "faixa_itens"))
    grandes = dados[dados["itens"] >= 40]
    comuns = dados[dados["itens"] <= 12]
    print(f"  pedidos com 40 itens ou mais: {len(grandes)}, de {grandes['itens'].min()} a {grandes['itens'].max()} itens;"
          f" o maior dos outros tem {comuns['itens'].max()}")
    print(f"  valor por item: {br((grandes['valor'] / grandes['itens']).median())} na mediana dos {len(grandes)},"
          f" {br((comuns['valor'] / comuns['itens']).median())} nos outros")

    print(f"\n6. O modelo da loja sem os {len(grandes)} pedidos de empresa ({len(comuns)} pedidos)")
    sem = ajustar(comuns, MODELO_LOJA)
    coeficientes(sem)
    print("  resíduo médio por faixa de itens")
    tabela(residuos_por(sem, comuns, "faixa_itens"))
    print("  resíduo médio por faixa de itens e categoria")
    cruzada(sem, comuns)

    print("\n7. Com a inclinação dos itens por categoria")
    final = ajustar(comuns, MODELO_FINAL)
    coeficientes(final)
    print("  resíduo médio por faixa de itens e categoria")
    cruzada(final, comuns)

    print("\n8. O coeficiente do cupom em cada modelo")
    caminho = [
        ("só o cupom, 1500 pedidos", dados, ["cupom"]),
        ("cupom e tipo de cliente, 1500", dados, ["cupom", "cliente_novo"]),
        ("modelo da loja, 1500", dados, MODELO_LOJA),
        (f"modelo da loja, {len(comuns)}", comuns, MODELO_LOJA),
        (f"itens por categoria, {len(comuns)}", comuns, MODELO_FINAL),
    ]
    for nome, base, colunas in caminho:
        print(f"  {nome:<34}{br(ajustar(base, colunas).coeficientes['cupom']):>8}")

    print(f"\n9. Onde caem 95% dos coeficientes em 500 sorteios dos {len(comuns)} pedidos")
    faixa = intervalo(comuns, MODELO_FINAL)
    for nome in ["cliente_novo", "cupom", "canal_app"]:
        print(f"  {nome:<14}{br(final.coeficientes[nome]):>8}   de {br(faixa.loc[nome, 'de']):>7}"
              f" até {br(faixa.loc[nome, 'até']):>7}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as erro:
        print(f"erro: {erro}", file=sys.stderr)
        sys.exit(1)
