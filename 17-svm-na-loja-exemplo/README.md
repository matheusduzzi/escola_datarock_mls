# 17. SVM na Loja Exemplo

Os arquivos da aula «SVM na Loja Exemplo» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `svm_loja.py` | As contas da aula: as dobras, a escala, os 15 SVM e a logística, a escolha, o custo contado, o custo medido e junho. Também é o módulo que o exercício usa. |
| `meu_svm.py` | Onde você escreve as funções `padronizar` e `no_teto` do exercício. Ele já compara o valor em reais e em centavos e o teto de 1 em 4 com o de 1 em 2. |
| `testar_meu_svm.py` | Os testes das suas duas funções, com casos pequenos e os números da parte 2 da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão, as respostas e a ficha do modelo. |

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso. Essa aula lê `dados/loja/pedidos_export_completo.csv`, na raiz do kit. Ela não usa o scikit-learn: o `svm_loja.py` treina o SVM e a logística no NumPy.

```bash
python3 svm_loja.py
```

Ele leva uns dez segundos e tem sete partes. As partes 1 a 5 e a 7 saem iguais toda vez e são as da aula. A parte 6 mede o tempo no seu computador: os números mudam a cada rodada.

```bash
python3 testar_meu_svm.py
python3 meu_svm.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError: escreva padronizar(treino, outro)`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 20 testes passaram`. Se um teste falhar, a mensagem diz qual caso deu errado e o que ele esperava.

Se aparecer `não achei`, você rodou fora da pasta da aula ou a pasta `dados/` não está na raiz do kit. Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

Depois do exercício, abra o `gerar.py` da pasta `dados/loja/`, na raiz do kit. Ele mostra como a chance de cancelar foi montada, coluna por coluna.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
