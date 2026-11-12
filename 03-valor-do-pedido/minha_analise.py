"""Exercício da aula «Valor do pedido»: as visitas antes do pedido.

Rode de dentro da pasta da aula:  python3 minha_analise.py
Escreva o seu código embaixo da linha marcada. As respostas vão no minhas-respostas.md.
"""
from regressao import ajustar, br, carregar, intervalo, residuos_por  # noqa: F401

MODELO_AULA = ["itens", "categoria", "cliente_novo", "cupom", "canal", "itens_por_categoria"]

dados = carregar()
comuns = dados[dados["itens"] <= 12]
print(f"{len(comuns)} pedidos de varejo")

# Escreva a sua análise daqui para baixo.
