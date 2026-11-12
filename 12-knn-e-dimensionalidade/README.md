# 12. KNN e dimensionalidade

Os arquivos da aula «KNN e dimensionalidade» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `vizinhos.py` | As contas da aula: a compra 6912 à mão com duas colunas, os vizinhos com as cinco colunas, as compras de junho com K diferentes, as colunas de ruído e o cubo. Também é o módulo que o exercício usa. |
| `meus_vizinhos.py` | Onde você escreve as funções `distancias` e `prever` do exercício. Ele já roda as compras de junho com a hora de dois jeitos. |
| `testar_meus_vizinhos.py` | Os testes das suas duas funções, com a compra 6912 da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão e as respostas do exercício. |

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso. Essa aula lê `dados/fraude/transacoes.csv`, na raiz do kit, e não usa o scikit-learn.

```bash
python3 vizinhos.py
```

Ele leva uns dez segundos. A saída é a mesma das seções 2 a 8 da aula e sai igual toda vez: as colunas de ruído da parte 5 vêm de um gerador de números aleatórios com ponto de partida fixo.

```bash
python3 testar_meus_vizinhos.py
python3 meus_vizinhos.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError: escreva distancias(conhecidas, nova, desvio)`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 12 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual a aula tem.

Se aparecer `não achei`, você rodou fora da pasta da aula ou a pasta `dados/` não está na raiz do kit. Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
