# 01. Regressão linear e MSE

Os arquivos da aula «Regressão linear e MSE» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `reta_e_erro.py` | As contas da aula com os quatro pedidos: o resíduo e o MSE das duas retas, a reta plana, o MSE de várias retas, o que acontece quando um número muda, a melhor reta pela conta das médias e o mesmo resultado no NumPy. |
| `minha_reta.py` | Onde você escreve as funções `mse` e `melhor_reta` do exercício. Os cinco pedidos da Bia já estão nele. |
| `testar_minha_reta.py` | Os testes das suas duas funções, com os quatro pedidos da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão e as respostas do exercício. |

## Como rodar

Você precisa do Python com o NumPy, como no README do kit do curso. Essa aula não usa o pandas nem os arquivos de `dados/`.

```bash
python3 reta_e_erro.py
```

A saída é a mesma da seção 9 da aula. Ela sai igual toda vez: não tem sorteio.

```bash
python3 testar_minha_reta.py
python3 minha_reta.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError` e a mensagem "escreva" seguida do nome da função. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 8 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual a aula tem.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
