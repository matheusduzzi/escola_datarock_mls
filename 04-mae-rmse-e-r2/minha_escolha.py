"""Exercício da aula «MAE, RMSE e R²»: o erro por categoria.

Rode de dentro da pasta da aula:  python3 minha_escolha.py
Escreva o seu código embaixo da linha marcada. As respostas vão no minhas-respostas.md.
"""
from metricas import dividir_por_data, erro_medio, mae, medir, prever, r2, rmse  # noqa: F401
from regressao import ajustar, br, carregar

MODELO_FINAL = ["itens", "categoria", "cliente_novo", "cupom", "canal", "itens_por_categoria"]

dados = carregar()
varejo = dados[dados["itens"] <= 12]
treino, teste = dividir_por_data(varejo)
ajuste = ajustar(treino, MODELO_FINAL)
previsto = prever(ajuste, teste, MODELO_FINAL)
print(f"{len(treino)} pedidos de treino, {len(teste)} de teste")

# Escreva a sua análise daqui para baixo.
