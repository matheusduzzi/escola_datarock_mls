# Minhas respostas: quatro hipóteses sobre os pedidos de junho

## Tarefa 1. À mão

| Pedido | Itens | Valor pago | Itens menos a média | Valor menos a média | Produto |
|---|---|---|---|---|---|
| P1 | 2 | 40 | | | |
| P2 | 4 | 50 | | | |
| P3 | 6 | 80 | | | |
| P4 | 8 | 90 | | | |
| P5 | 10 | 90 | | | |

Média dos itens: | Média do valor:

Variância dos itens (conta):

Covariância com o valor (conta):

| Ajuste | Coeficiente por item (conta) | Intercepto (conta) |
|---|---|---|
| sem penalidade | | |
| Ridge alfa 8 | | |
| Lasso alfa 32 | | |

A partir de que alfa o Lasso zera os itens:

Em caixas: variância , covariância

| Ajuste | Coeficiente por caixa | Em R$ por item |
|---|---|---|
| sem penalidade | | |
| Ridge alfa 8 | | |
| Lasso alfa 32 | | |

O que mudou e por quê:

Ridge alfa 8 padronizado: por desvio , em R$ por item . Em caixas dá o mesmo?

## Tarefa 2. No código

Saída de `python3 testar_minha_comparacao.py`:

```
```

Saída de `python3 minha_comparacao.py`:

```
```

## Tarefa 3. As hipóteses

| Palpite do Caio | Apoia, não apoia ou não decide | O número da saída que mostra | Penalização que supõe esse palpite |
|---|---|---|---|
| 1. Itens e peso contam a mesma história | | | |
| 2. Prazo, hora e dia não mudam o valor | | | |
| 3. O cupom não muda o valor | | | |
| 4. Com a penalidade, o coeficiente dos itens fica perto do certo | | | |

O que um zero do Lasso quer dizer (palpite 3):

Para onde a penalidade puxa o coeficiente (palpite 4):

Qual penalização eu levaria para a aula «Escolher regularização», e por quê:
