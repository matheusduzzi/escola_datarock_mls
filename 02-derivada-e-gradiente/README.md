# 02. Derivada e gradiente

Os arquivos da aula «Derivada e gradiente» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `gradiente.py` | As contas da aula com os quatro pedidos: o erro da reta de partida, a derivada por mexidas pequenas e pela regra, um passo com taxa 0,01, trinta passos com três taxas, a reta de menor erro e os itens contados a partir da média. |
| `minha_atualizacao.py` | Onde você escreve as funções `gradiente` e `passo` do exercício. Os cinco pedidos da Papelaria já estão nele. |
| `testar_minha_atualizacao.py` | Os testes das suas duas funções, com os quatro pedidos da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão e as respostas do exercício. |

## Como rodar

Você precisa do Python com o NumPy, como no README do kit do curso. Essa aula não usa o pandas nem os arquivos de `dados/`.

```bash
python3 gradiente.py
```

A saída é a mesma da seção 7 da aula. Ela sai igual toda vez: não tem sorteio.

```bash
python3 testar_minha_atualizacao.py
python3 minha_atualizacao.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError: escreva gradiente(a, b, x, y)`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 8 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual a aula tem.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
