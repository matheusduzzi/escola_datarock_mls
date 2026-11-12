# 10. Ridge, Lasso e Elastic Net

Os arquivos da aula «Ridge, Lasso e Elastic Net» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `penalizacao.py` | As contas da aula: a regressão nos 40 primeiros pedidos de janeiro, a conta à mão com cinco pedidos, a padronização, o Ridge, o Lasso, o Elastic Net e o peso em três unidades. Também é o módulo que o exercício usa. |
| `minha_comparacao.py` | Onde você escreve as funções `padronizar` e `para_reais` do exercício. Ele já monta as tabelas dos 40 primeiros pedidos de junho. |
| `testar_minha_comparacao.py` | Os testes das suas duas funções, com os pedidos da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão e as respostas do exercício. |

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso. Essa aula lê `dados/valor/pedidos_valor.csv`, na raiz do kit, e não usa o scikit-learn.

```bash
python3 penalizacao.py
```

A saída é a mesma das seções 2 a 8 da aula. Ela sai igual toda vez: não tem sorteio.

```bash
python3 testar_minha_comparacao.py
python3 minha_comparacao.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError: escreva padronizar(treino, outros)`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 9 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual a aula tem.

Se aparecer `não achei`, você rodou fora da pasta da aula ou a pasta `dados/` não está na raiz do kit. Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
