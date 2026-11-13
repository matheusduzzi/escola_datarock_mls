"""As contas da aula «MAE, RMSE e R²», nos pedidos entregues da Loja Exemplo.

    python3 medir_erro.py

Lê ../dados/valor/pedidos_valor.csv e mostra, em ordem: o R² do modelo medido nos próprios pedidos do
ajuste, os mesmos erros em três réguas, a divisão por data, quatro modelos medidos em maio e junho, o
mesmo com os pedidos de empresa, o erro por tipo de cliente e o R² ajustado com colunas sorteadas.
Sai igual toda vez.
"""
import sys

import numpy as np

from metricas import (com_colunas_sorteadas, dividir_por_data, mae, medir, prever, r2, r2_ajustado, rmse,
                      treinar_e_medir)
from regressao import ajustar, br, carregar

MODELO_LOJA = ["itens", "categoria", "cliente_novo", "cupom", "canal"]
MODELO_FINAL = MODELO_LOJA + ["itens_por_categoria"]


def linha(nome, m, largura=26, r2_casas=3):
    print(f"  {nome:<{largura}}{br(m['MAE']):>9}{br(m['RMSE']):>9}{br(m['R²'], r2_casas):>9}")


def cabecalho(largura=26):
    print(f"  {'':<{largura}}{'MAE':>9}{'RMSE':>9}{'R²':>9}")


def main():
    dados = carregar()
    varejo = dados[dados["itens"] <= 12]

    print("1. O modelo com itens por categoria, medido nos mesmos 1500 pedidos do ajuste")
    caio = ajustar(dados, MODELO_FINAL)
    m = medir(dados["valor"], caio.previsto)
    print(f"  R² {br(m['R²'], 3)}   MAE {br(m['MAE'])}   RMSE {br(m['RMSE'])}")

    print("\n2. Os mesmos quatro pedidos, dois jeitos de errar (erros em reais)")
    real = np.array([100.0, 100.0, 100.0, 100.0])
    for nome, erros in [("A: 10, 10, 10, 10", [10, 10, 10, 10]), ("B: 0, 0, 0, 40", [0, 0, 0, 40])]:
        previsto = real - np.array(erros, dtype=float)
        mse = rmse(real, previsto) ** 2
        print(f"  {nome:<20} MAE {br(mae(real, previsto)):>6}   MSE {br(mse):>7}   RMSE {br(rmse(real, previsto)):>6}")

    print("\n3. Treino e teste pela data, nos pedidos de varejo (até 12 itens)")
    treino, teste = dividir_por_data(varejo)
    print(f"  treino  {len(treino)} pedidos, de {treino['data_pedido'].min()} a {treino['data_pedido'].max()}")
    print(f"  teste   {len(teste)} pedidos, de {teste['data_pedido'].min()} a {teste['data_pedido'].max()}")

    print("\n4. Quatro modelos ajustados no treino e medidos no teste (varejo)")
    cabecalho()
    media = np.full(len(teste), treino["valor"].mean())
    linha("média do treino", medir(teste["valor"], media))
    for nome, colunas in [("só itens", ["itens"]), ("modelo da loja", MODELO_LOJA),
                          ("itens por categoria", MODELO_FINAL)]:
        _, no_treino, no_teste = treinar_e_medir(treino, teste, colunas)
        linha(nome, no_teste)
    _, final_treino, final_teste = treinar_e_medir(treino, teste, MODELO_FINAL)
    print(f"  itens por categoria no próprio treino: MAE {br(final_treino['MAE'])}   RMSE {br(final_treino['RMSE'])}"
          f"   R² {br(final_treino['R²'], 3)}")

    print("\n5. O mesmo, com os 12 pedidos de empresa (todos os 1500)")
    treino_t, teste_t = dividir_por_data(dados)
    grandes_treino = int((treino_t["itens"] >= 40).sum())
    grandes_teste = int((teste_t["itens"] >= 40).sum())
    print(f"  treino {len(treino_t)} pedidos ({grandes_treino} de empresa)   teste {len(teste_t)} pedidos"
          f" ({grandes_teste} de empresa)")
    cabecalho()
    for nome, colunas in [("modelo da loja", MODELO_LOJA), ("itens por categoria", MODELO_FINAL)]:
        _, _, no_teste = treinar_e_medir(treino_t, teste_t, colunas)
        linha(nome, no_teste)
    ajuste_t = ajustar(treino_t, MODELO_FINAL)
    previsto_t = prever(ajuste_t, teste_t, MODELO_FINAL)
    erro_t = teste_t["valor"] - previsto_t
    pior = erro_t.abs().idxmax()
    p = teste_t.loc[pior]
    print(f"  maior erro: pedido {p['id_pedido']}, {p['itens']} itens de {p['categoria']},"
          f" pago {br(p['valor'])}, previsto {br(previsto_t[pior])}, erro {br(erro_t[pior])}")
    print(f"  esse pedido é {br(100 * erro_t[pior] ** 2 / (erro_t ** 2).sum(), 1)}% da soma dos erros ao quadrado")
    c = ajuste_t.coeficientes
    print(f"  cada item de Eletro soma {br(c['itens'])} + {br(c['itens_Eletro'])} = {br(c['itens'] + c['itens_Eletro'])}"
          f" ao previsto")
    loja_t = prever(ajustar(treino_t, MODELO_LOJA), teste_t, MODELO_LOJA)
    print(f"  o modelo da loja previu {br(loja_t[pior])} para esse pedido, erro {br(teste_t['valor'][pior] - loja_t[pior])}")
    sem = teste_t.index != pior
    print(f"  sem esse pedido no teste: MAE {br(mae(teste_t['valor'][sem], previsto_t[sem]))}"
          f"   RMSE {br(rmse(teste_t['valor'][sem], previsto_t[sem]))}"
          f"   R² {br(r2(teste_t['valor'][sem], previsto_t[sem]), 3)}")

    print("\n6. Itens por categoria no teste: por tipo de cliente e os erros grandes")
    previsto = prever(ajustar(treino, MODELO_FINAL), teste, MODELO_FINAL)
    print(f"  {'':<14}{'pedidos':>8}{'MAE':>9}{'RMSE':>9}{'R²':>9}{'desvio do valor':>17}")
    for valor_tipo, nome in [(0, "recorrentes"), (1, "novos")]:
        k = teste["cliente_novo"] == valor_tipo
        m = medir(teste["valor"][k], previsto[k])
        print(f"  {nome:<14}{int(k.sum()):>8}{br(m['MAE']):>9}{br(m['RMSE']):>9}{br(m['R²'], 3):>9}"
              f"{br(teste['valor'][k].std(ddof=0)):>17}")

    erro = teste["valor"] - previsto
    grandes = erro.abs() > 150
    print(f"  pedidos com erro acima de 150 reais, para mais ou para menos: {int(grandes.sum())} de {len(teste)},"
          f" {br(100 * (erro[grandes] ** 2).sum() / (erro ** 2).sum(), 1)}% da soma dos erros ao quadrado")
    por_categoria = teste.loc[grandes, "categoria"].value_counts()
    print("  desses, por categoria: " + ", ".join(f"{c} {n}" for c, n in por_categoria.items())
          + f"; com 4 itens ou mais: {int((teste.loc[grandes, 'itens'] >= 4).sum())}")

    print("\n7. Colunas sorteadas, sem relação com o valor, somadas ao modelo com itens por categoria")
    print(f"  {'sorteadas':<10}{'colunas':>8}{'R² treino':>11}{'ajustado':>10}{'R² teste':>10}{'MAE teste':>11}")
    com_sorteio, nomes = com_colunas_sorteadas(varejo, 40)
    treino_s, teste_s = dividir_por_data(com_sorteio)
    for quantas in [0, 10, 40]:
        colunas = MODELO_FINAL + nomes[:quantas]
        ajuste, no_treino, no_teste = treinar_e_medir(treino_s, teste_s, colunas)
        k = len(ajuste.coeficientes) - 1
        print(f"  {quantas:<10}{k:>8}{br(no_treino['R²'], 4):>11}{br(r2_ajustado(no_treino['R²'], len(treino_s), k), 4):>10}"
              f"{br(no_teste['R²'], 4):>10}{br(no_teste['MAE']):>11}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as erro:
        print(f"erro: {erro}", file=sys.stderr)
        sys.exit(1)
