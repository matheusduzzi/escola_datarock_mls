# Minhas respostas: a mensagem que repete e as tentativas do cartão

## Tarefa 1. À mão

Mensagem: "Quero trocar o pedido. Dá para trocar?" Palavras: `trocar` duas vezes, `pedido` uma vez, `chegou` não aparece. P(atraso) = 537 ÷ 936, P(troca) = 399 ÷ 936, alpha = 1.

### Multinomial

| Fila | Total de palavras (3 palavras) | P(pedido \| fila) | P(trocar \| fila) | Produto (conta) | Parte na soma |
|---|---|---|---|---|---|
| atraso | | | | | |
| troca | | | | | |

### Bernoulli

| Fila | P(pedido aparece) | P(trocar aparece) | P(chegou falta) | Produto (conta) | Parte na soma |
|---|---|---|---|---|---|
| atraso | | | | | |
| troca | | | | | |

Para o Diego: para que fila cada variante manda, e o que cada uma usou que a outra não usou:

Por que o Bernoulli muda de fila com as 140 palavras:

## Tarefa 2. No código

Saída de `python3 testar_meu_bayes.py`:

```
```

Saída de `python3 meu_bayes.py`:

```
```

## Tarefa 3. Ler e escolher

1. Por que a curva normal não combina com as tentativas (dois números da parte A):

2. O que muda em junho com as tentativas como categoria, e se a Renata pode escolher por esse resultado:

3. O que a coluna dos dias desde a abertura fez, e por quê:

4. A escolha da variante:

| Como o dado está | Variante | O que ela supõe | O que eu olharia antes de usar |
|---|---|---|---|
| mensagens do chat como contagem de cada palavra | | | |
| as mesmas mensagens como "a palavra aparece ou não" | | | |
| e-mails com a conversa inteira colada, de 200 a 2.000 palavras | | | |
| canal da compra, app ou site | | | |
| tentativas do cartão, de 1 a 6 | | | |
| valor da compra em reais | | | |

Para a Renata, como escolher a variante e o que fazer antes de confiar na probabilidade:
