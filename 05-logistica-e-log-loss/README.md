# 05. Logística e log-loss

Os arquivos da aula «Logística e log-loss» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `logistica.py` | As contas da aula: a nota, a perda e o gradiente nos quatro pedidos, um passo, e os passos nos 1155 pedidos de janeiro a março. |
| `minha_perda.py` | Onde você escreve as funções `sigmoide`, `log_loss` e `gradiente` do exercício. Os seis pedidos da Bia já estão nele. |
| `testar_minha_perda.py` | Os testes das suas três funções, com os quatro pedidos da aula. |
| `minhas-respostas.md` | Onde você escreve as contas da calculadora e as respostas do exercício. |

Os dois programas leem `../dados/loja/pedidos.csv`, a base de pedidos com cancelamento do kit do curso. Eles usam só os pedidos de janeiro a março.

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso.

```bash
python3 logistica.py
```

A saída é a mesma da seção 8 da aula. Ela sai igual toda vez: não tem sorteio.

```bash
python3 testar_minha_perda.py
python3 minha_perda.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError`: o teste com `escreva gradiente(a, b, x, y)`, e o `minha_perda.py` com `escreva sigmoide(z)`. É o esperado. Quando as três funções estiverem certas, o teste mostra `ok: 11 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual a aula tem.

Se aparecer `No such file or directory: '../dados/loja/pedidos.csv'`, você está fora da pasta da aula ou baixou só essa pasta, sem a `dados/` do kit. Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
