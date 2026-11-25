# 14. Naive Bayes

Os arquivos da aula «Naive Bayes» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `bayes.py` | As contas da aula: a mensagem «pix não chegou» à mão nas duas variantes de texto, a suavização, a mensagem com dois assuntos, a frase repetida, as compras com a curva normal, em log, a correlação dentro de cada classe e o canal como categoria. Também é o módulo que o exercício usa. |
| `meu_bayes.py` | Onde você escreve a função `categorica_log` do exercício. Ele já roda as compras de junho. |
| `testar_meu_bayes.py` | Os testes da sua função, com o canal da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão, a saída e as respostas do exercício. |

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso. Essa aula lê `dados/mensagens/mensagens.csv` e `dados/fraude/transacoes.csv`, na raiz do kit, e não usa o scikit-learn.

```bash
python3 bayes.py
```

Ele roda em menos de um segundo. A saída tem dez partes, as mesmas da aula, e sai igual toda vez.

```bash
python3 testar_meu_bayes.py
python3 meu_bayes.py
```

Antes de você escrever a função, os dois param com `NotImplementedError: escreva categorica_log(treino, y, novos, alpha)`. É o esperado. Quando a função estiver certa, o teste mostra `ok: 12 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual era o certo.

Se aparecer `não achei`, você rodou fora da pasta da aula ou a pasta `dados/` não está na raiz do kit. Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
