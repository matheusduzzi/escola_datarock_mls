# 15. Naive Bayes na prática

Os arquivos da aula «Naive Bayes na prática» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `filas.py` | As contas da aula: a fila maior, a regra de palavras do Diego e o Naive Bayes, escolhidos nos meses de fevereiro a maio, junho aberto uma vez, os erros da regra, as mensagens de julho e o arquivo de previsões. Também é o módulo que o exercício usa. |
| `julho.csv` | As 40 mensagens de julho que o Diego separou e marcou, 10 de cada fila. |
| `minha_analise.py` | Onde você escreve as funções `matriz_confusao` e `cobertura` do exercício. Ele já roda julho. |
| `testar_minha_analise.py` | Os testes das suas funções. |
| `minhas-respostas.md` | Onde você escreve a saída e as respostas do exercício. |

## Como rodar

Você precisa do Python 3.9 ou mais novo. Essa aula só usa a biblioteca padrão: não precisa do pandas nem do scikit-learn. Ela lê `dados/mensagens/mensagens.csv`, na raiz do kit, e o `julho.csv` dessa pasta.

```bash
python3 filas.py
```

Ele roda em menos de um segundo. A saída tem oito partes, as mesmas da aula, e sai igual toda vez. A parte 8 grava `saida/previsoes.csv` nessa pasta e mostra a impressão digital do arquivo: `4beabef8417a`. Se a sua for diferente, algum arquivo da pasta ou dos dados mudou.

```bash
python3 testar_minha_analise.py
python3 minha_analise.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError: escreva matriz_confusao(reais, previstos, assuntos)`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 14 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual era o certo.

Se aparecer `não achei`, você rodou fora da pasta da aula ou a pasta `dados/` não está na raiz do kit. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
