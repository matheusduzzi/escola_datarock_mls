# Kit do curso ML supervisionado

Os arquivos que você usa nas aulas: o diagnóstico, as bases da Loja Exemplo e as reservas do case de viagens.

## Como o kit está organizado

| Pasta | O que tem |
|---|---|
| `00-apresentacao/` | O [diagnóstico](00-apresentacao/diagnostico.md), os dados e programas das tarefas, a [folha de respostas](00-apresentacao/minhas-respostas.md), a [folha da ordem de estudo](00-apresentacao/minha-ordem.md) e a `primeira_olhada.py` da aula 00. |
| `dados/valor/` | 1500 pedidos entregues com o valor pago, para as aulas de regressão. Ver [dados/valor/LEIA-ME.md](dados/valor/LEIA-ME.md). |
| `dados/loja/` | 2400 pedidos com o rótulo de cancelamento, os mesmos do curso ML: preparação e validação. Ver [dados/loja/LEIA-ME.md](dados/loja/LEIA-ME.md). |
| `dados/fraude/` | 8000 compras com cartão, com o rótulo de fraude. Ver [dados/fraude/LEIA-ME.md](dados/fraude/LEIA-ME.md). |
| `dados/mensagens/` | 1800 mensagens de clientes ao atendimento, com o assunto. Ver [dados/mensagens/LEIA-ME.md](dados/mensagens/LEIA-ME.md). |
| `dados/viagens/` | 3600 reservas da Agência Exemplo, para o case final. Ver [dados/viagens/LEIA-ME.md](dados/viagens/LEIA-ME.md). |
| `NN-titulo/` | Uma pasta por aula, com o número e o título da aula. Elas entram no kit junto com as aulas. |

Os comandos das aulas rodam de dentro da pasta da aula. Quando uma aula usa uma base compartilhada, ela diz o caminho, por exemplo `../dados/valor/pedidos_valor.csv`.

## Do que você precisa

- Python 3.9 ou mais novo, com o pandas e o NumPy. É o mesmo ambiente do curso ML: preparação e validação; se você ainda não preparou o seu, a aula 38. Dependências e ambientes, de Python para dados, mostra como.
- Um editor de código e um terminal.

O scikit-learn entra na primeira aula que ajusta um modelo com ele, e o matplotlib na primeira que desenha um gráfico. Cada uma traz a versão, o Python mínimo e o comando de instalação.

Na pasta do kit, instale o que o diagnóstico usa:

```bash
python3 -m pip install -r requirements.txt
```

E veja se funcionou:

```bash
cd 00-apresentacao
python3 tarefa3.py
```

A saída tem quatro linhas. A primeira é `média usada: 229.69`; as outras três são a tarefa 3 do diagnóstico. Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

Para gerar de novo as bases compartilhadas, rode `python3 gerar.py` dentro de cada pasta de `dados/`. As saídas são:

| Pasta | Saída de `python3 gerar.py` |
|---|---|
| `dados/valor/` | `pedidos=1500 com_cupom=436 clientes_novos=564 valor_total=320875.21` |
| `dados/loja/` | `pedidos=2400 cancelados=467 cancelados_com_entrega=22 entregues_com_dias_zero=0` |
| `dados/fraude/` | `transacoes=8000 fraudes=101` |
| `dados/mensagens/` | `mensagens=1800 atraso=669 troca=495 pagamento=362 outro=274` |
| `dados/viagens/` | `reservas=3600 canceladas=736 flexiveis=993` |

Os geradores só usam a biblioteca padrão e saem iguais toda vez.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.

## Licença

A licença de uso e redistribuição do kit ainda vai ser definida pela Escola Datarock. Até lá, use os arquivos para estudar o curso. A Loja Exemplo e a Agência Exemplo são inventadas, e todos os dados foram gerados pelos programas do próprio kit.

## Aulas com arquivos neste kit

| Pasta | Aula |
|---|---|
| [00-apresentacao/](00-apresentacao/) | Apresentação |
| [01-regressao-linear-e-mse/](01-regressao-linear-e-mse/) | Regressão linear e MSE |
| [02-derivada-e-gradiente/](02-derivada-e-gradiente/) | Derivada e gradiente |
| [03-valor-do-pedido/](03-valor-do-pedido/) | Valor do pedido |
| [04-mae-rmse-e-r2/](04-mae-rmse-e-r2/) | MAE, RMSE e R² |
| [05-logistica-e-log-loss/](05-logistica-e-log-loss/) | Logística e log-loss |
| [06-odds-e-fronteira/](06-odds-e-fronteira/) | Odds e fronteira |
| [07-cancelamento-de-pedidos/](07-cancelamento-de-pedidos/) | Cancelamento de pedidos |
| [08-metricas-e-prevalencia/](08-metricas-e-prevalencia/) | Métricas e prevalência |
| [09-calibracao-e-limiar/](09-calibracao-e-limiar/) | Calibração e limiar |
| [10-ridge-lasso-e-elastic-net/](10-ridge-lasso-e-elastic-net/) | Ridge, Lasso e Elastic Net |
| [11-escolher-regularizacao/](11-escolher-regularizacao/) | Escolher regularização |
| [12-knn-e-dimensionalidade/](12-knn-e-dimensionalidade/) | KNN e dimensionalidade |
| [13-knn-em-fraude-ficticia/](13-knn-em-fraude-ficticia/) | KNN em fraude fictícia |
| [15-naive-bayes-na-pratica/](15-naive-bayes-na-pratica/) | Naive Bayes na prática |
| [16-svm-e-kernels/](16-svm-e-kernels/) | SVM e kernels |
| [17-svm-na-loja-exemplo/](17-svm-na-loja-exemplo/) | SVM na Loja Exemplo |
| [18-case-de-viagens/](18-case-de-viagens/) | Case de viagens |
