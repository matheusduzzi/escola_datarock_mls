# Minhas respostas: a regra do boleto

## Tarefa 1. Na calculadora

Regra: score = −2 + 1 × boleto.

Conta da sigmoide para um pedido com boleto:

Conta da sigmoide para um pedido sem boleto:

| Pedido | Boleto | Cancelou? | Score | Nota | Nota para o que aconteceu | Perda | Nota − cancelou |
|---|---|---|---|---|---|---|---|
| 15 | 1 | sim | | | | | |
| 57 | 1 | não | | | | | |
| 12 | 0 | sim | | | | | |
| 14 | 0 | não | | | | | |
| 18 | 0 | não | | | | | |
| 26 | 0 | não | | | | | |
| Soma | | | | | | | |

Log-loss da regra:

Log-loss da nota 0,2 para todos (conta):

Qual ganha nesses seis pedidos?

Derivada no −2 (conta):

Derivada no 1 (conta):

Passo com taxa 1:

- a novo:
- b novo:
- regra nova:

Log-loss da regra nova nos seis pedidos (conta):

Caiu?

## Tarefa 2. No NumPy

Saída de `python3 testar_minha_perda.py`:

```
```

Saída de `python3 minha_perda.py`:

```
```

Bateu com a tarefa 1? Se não bateu, onde estava a diferença?

## Tarefa 3. A resposta para a Bia

