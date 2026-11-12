# 18. Case de viagens

Os arquivos da aula «Case de viagens» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `case_viagens.py` | As contas da aula: a divisão, o preço, a coluna que nasce depois, os oito candidatos nas dobras, a escolha, o custo, 2026 e os erros do escolhido. Também é o módulo que o exercício usa. |
| `minha_comparacao.py` | Onde você escreve as funções `treino_da_dobra` e `economia_da_lista` do exercício. Ele já compara os candidatos com a ligação a R$ 35 e três tamanhos de lista. |
| `testar_minha_comparacao.py` | Os testes das suas duas funções, com casos pequenos e os números das partes 1 e 4 da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão, as respostas e a defesa. |

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso. Essa aula lê `dados/viagens/reservas.csv`, na raiz do kit. Ela não usa o scikit-learn: o `case_viagens.py` treina a logística, o Naive Bayes, o KNN e o SVM no NumPy.

```bash
python3 case_viagens.py
```

Ele leva uns cinco segundos e tem oito partes. Todas saem iguais toda vez, menos a coluna de segundos da parte 6, que mede o tempo no seu computador.

```bash
python3 testar_minha_comparacao.py
python3 minha_comparacao.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 21 testes passaram`. Se um teste falhar, a mensagem diz qual caso deu errado e o que ele esperava.

Se aparecer `não achei`, você rodou fora da pasta da aula ou a pasta `dados/` não está na raiz do kit. Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

Depois do exercício, abra o `gerar.py` da pasta `dados/viagens/`, na raiz do kit. Ele mostra como a chance de cancelar foi montada, coluna por coluna. Compare com a sua defesa.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
