# 16. SVM e kernels

Os arquivos da aula «SVM e kernels» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `fronteiras.py` | As contas da aula: a reta de maior margem nos dez pedidos, as três retas, abril, a margem flexível, os três kernels no mapa e o kernel por dentro. Também é o módulo que o exercício usa. |
| `entregas.csv` | Os 120 pedidos entregues de motoboy em março e abril, com o endereço em km a partir do centro de distribuição (`leste_km`, `norte_km`) e o rótulo `cancelado`. |
| `gerar_entregas.py` | O programa que gera o `entregas.csv`. Abra só depois de fazer a aula e o exercício. |
| `meu_kernel.py` | Onde você escreve as funções `kernel_rbf` e `decisao` do exercício. Ele já compara os valores de gamma e treina a reta dos dez sem um pedido. |
| `testar_meu_kernel.py` | Os testes das suas duas funções, com os números da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão e as respostas do exercício. |

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso. Essa aula lê `dados/loja/pedidos.csv`, na raiz do kit, e o `entregas.csv` dessa pasta. Ela não usa o scikit-learn: o `fronteiras.py` treina o SVM no NumPy.

```bash
python3 fronteiras.py
```

Ele leva uns dois segundos. A saída é a mesma das seções 2 a 11 da aula e sai igual toda vez.

```bash
python3 testar_meu_kernel.py
python3 meu_kernel.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError: escreva kernel_rbf(A, B, gamma)`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 15 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual a aula tem.

Para gerar de novo o `entregas.csv`, rode `python3 gerar_entregas.py`. A saída é `pedidos=120 marco=80 abril=40 cancelados=44`, e o arquivo sai igual, byte a byte.

Se aparecer `não achei`, você rodou fora da pasta da aula ou a pasta `dados/` não está na raiz do kit. Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
