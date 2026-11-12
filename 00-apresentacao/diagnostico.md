# Diagnóstico: por onde começar em ML supervisionado

A gerente da Loja Exemplo quer duas coisas para o segundo semestre. A primeira é uma estimativa do valor de cada pedido, para planejar estoque e frete. A segunda é um alerta de fraude nas compras com cartão. Antes de abrir qualquer modelo, ela me mandou três frases que ouviu da equipe: "a reta erra pouco porque os erros somam quase zero", "o modelo de fraude acerta 95%, então está ótimo" e "pedido com cupom vale menos, então o cupom faz o cliente gastar menos". Eu refiz as contas de cada frase. As cinco tarefas abaixo são essas contas e o código que eu usei.

Elas pedem o que o curso usa desde a primeira aula: uma reta e o erro dela, uma função do NumPy com teste, a preparação feita só no treino, uma métrica com classe rara e uma comparação dentro de grupos. O diagnóstico não é prova e não tem nota. Ele mostra por onde você começa. Anote na [folha de respostas](minhas-respostas.md) a conta que fez, o comando que rodou e a saída que apareceu. Só depois compare com as respostas.

As tarefas 1, 4 e 5 são de lápis e papel ou calculadora. As tarefas 2 e 3 rodam de dentro da pasta `00-apresentacao/`:

```bash
cd 00-apresentacao
python3 tarefa3.py
```

No Windows, troque `python3` por `py`.

## Tarefa 1. Uma reta e o erro dela

Alguém da equipe propôs uma regra para prever o valor de um pedido pela quantidade de itens: valor previsto = 40 + 25 × itens, em reais. Ela testou a regra em quatro pedidos:

| Pedido | Itens | Valor pago (R$) |
|---|---|---|
| A | 1 | 70 |
| B | 2 | 85 |
| C | 4 | 150 |
| D | 5 | 160 |

1. Qual o valor previsto de cada pedido?
2. Qual a diferença entre o valor pago e o previsto em cada um? Essa diferença se chama resíduo.
3. Some os quatro resíduos. Depois eleve cada um ao quadrado e tire a média.
4. Outra pessoa propôs valor previsto = 30 + 27 × itens. Faça as mesmas contas. Qual das duas regras erra menos pela média dos quadrados?
5. A equipe disse "a reta erra pouco porque os erros somam quase zero". Por que a soma dos resíduos não serve para medir o erro?
6. Na primeira regra, o que o número 25 diz sobre os pedidos?

## Tarefa 2. Prever e medir com o NumPy

Em `tarefa2.py`, escreva duas funções, sem loop em Python:

| Função | O que retorna |
|---|---|
| `prever(a, b, x)` | `a + b * x` para cada valor de `x`, como array do NumPy; `x` pode chegar como lista |
| `erro_quadratico_medio(y, previsto)` | a média de `(y - previsto)` ao quadrado; para com `ValueError` se os dois tiverem tamanhos diferentes |

Depois rode:

```bash
python3 tarefa2.py
python3 testar_tarefa2.py
```

Enquanto as funções não existem, os dois comandos param com `NotImplementedError`. Anote na folha:

1. As duas linhas que `tarefa2.py` mostrou. Elas batem com a sua conta da tarefa 1?
2. O teste chama `erro_quadratico_medio([1, 2, 3, 4], [2])`. Sem a checagem de tamanho, o que o NumPy faz com o `[2]`? Por que isso é perigoso?
3. Por que o teste usa `abs(e - 104 / 3) < 1e-9` e não `e == 104 / 3`?

## Tarefa 3. Padronizar sem olhar o teste

Abra `tarefa3.py` e leia tudo antes de rodar. Os 30 primeiros pedidos de abril são o treino e os 10 últimos, da última semana do mês, são o teste. O modelo vai ser usado em maio. Rode e anote as quatro linhas.

1. Que pedidos entraram na média e no desvio usados para padronizar?
2. Por que isso é um problema, se o modelo vai prever pedidos de maio?
3. Reescreva o programa para calcular a média e o desvio só no treino e aplicar os mesmos dois números no treino e no teste. Rode e anote as quatro linhas novas.
4. Na versão nova, a média padronizada do treino dá zero e a do teste fica bem longe de zero. Isso é um erro do programa? O que esse número conta sobre a última semana de abril?

## Tarefa 4. Acertar 95% numa classe rara

Num mês com 20.000 compras com cartão, 1,5% são fraude. O modelo de fraude marca 80 de cada 100 fraudes. Entre as compras legítimas, ele marca 4 de cada 100 por engano.

1. Monte a tabela com quatro contagens: fraudes marcadas, fraudes que passaram, legítimas marcadas e legítimas que passaram.
2. Quantas compras o modelo marca no mês? De cada 100 marcadas, quantas são fraude?
3. Que porcentagem das 20.000 o modelo classifica certo? E quanto acerta quem diz "não é fraude" para toda compra?
4. A equipe consegue revisar 300 compras por mês. O que acontece com as outras marcadas?
5. Num mês em que só 0,5% são fraude, com o mesmo modelo, de cada 100 marcadas, quantas são fraude? O que mudou: o modelo ou o mês?

## Tarefa 5. O cupom e o tipo de cliente

A equipe separou mil pedidos por tipo de cliente e por cupom:

| Cliente | Cupom | Pedidos | Valor médio (R$) |
|---|---|---|---|
| novo | com cupom | 300 | 120 |
| novo | sem cupom | 100 | 110 |
| recorrente | com cupom | 100 | 260 |
| recorrente | sem cupom | 500 | 250 |

1. Qual o valor médio de todos os pedidos com cupom? E dos sem cupom?
2. Dentro dos clientes novos, qual a diferença entre com e sem cupom? E dentro dos recorrentes?
3. Por que a comparação geral e a comparação dentro de cada tipo de cliente dão sinais opostos?
4. A equipe disse "o cupom faz o cliente gastar menos". E se alguém disser "o cupom faz o cliente gastar R$ 10 a mais"? O que você precisaria saber sobre quem recebe cupom antes de concordar com qualquer uma das duas frases?

## Depois das tarefas

Compare a folha com as respostas na aula e veja a tabela da apresentação do curso. Ela diz que aula estudar para cada tarefa em que você travou.

Se precisar de ajuda, deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
