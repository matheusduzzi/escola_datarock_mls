# 06. Odds e fronteira

Os arquivos da aula «Odds e fronteira» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `odds.py` | As contas da aula: chance e odds na contagem, os três modelos, o b lido em odds, as bases diferentes e a fronteira de cada corte. |
| `minhas_odds.py` | Onde você escreve as funções `comparar` e `prazo_da_fronteira` do exercício. A regra da Lia, os pedidos A e B e o corte de 30% já estão nele. |
| `testar_minhas_odds.py` | Os testes das suas duas funções, com o modelo da aula. |
| `minhas-respostas.md` | Onde você escreve as contas da calculadora e a resposta para a Lia. |

Os programas leem `../dados/loja/pedidos.csv`, a base de pedidos com cancelamento do kit do curso. Eles usam só os pedidos de janeiro a março.

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso.

```bash
python3 odds.py
```

A saída é a mesma da seção 9 da aula e sai igual toda vez: o ajuste começa de zero e não tem sorteio. Leva uns 3 segundos.

```bash
python3 testar_minhas_odds.py
python3 minhas_odds.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError: escreva comparar(score_base, b)`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 19 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual a aula tem.

Se aparecer `No such file or directory: '../dados/loja/pedidos.csv'`, você está fora da pasta da aula ou baixou só essa pasta, sem a `dados/` do kit. Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
