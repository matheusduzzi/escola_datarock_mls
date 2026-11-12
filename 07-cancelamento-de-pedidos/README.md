# 07. Cancelamento de pedidos

Os arquivos da aula «Cancelamento de pedidos» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `pipeline_corrigido.py` | A preparação dos pedidos do curso ML: preparação e validação, o mesmo arquivo da aula «Detectar vazamento». Não mexa nele. |
| `classificador.py` | O classificador pronto do curso anterior, o mesmo da aula «Detectar vazamento». Aqui ele serve só para comparar. |
| `treino.py` | A regressão logística treinada pelo gradiente da log-loss, a conta da aula 05 com todas as colunas. |
| `cancelamento.py` | As contas da aula: a preparação, o treino passo a passo, os coeficientes, as matrizes de confusão nos cortes 0,5 e 0,333 e a preparação refeita do Rafael. |
| `meu_modelo.py` | Onde você escreve as funções `matriz`, `por_grupo`, `salvar` e `carregar` do exercício. A preparação e o treino já estão chamados nele. |
| `testar_meu_modelo.py` | Os testes das suas quatro funções, com seis pedidos inventados dentro do próprio teste. |
| `minhas-respostas.md` | Onde você escreve as contas à mão e as respostas do exercício. |

Os pedidos vêm de `../dados/loja/pedidos_export_completo.csv`, a base compartilhada do kit. Treino de janeiro a março, validação em abril e maio. Junho é o teste e sai logo na leitura.

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso. Não precisa do scikit-learn.

```bash
python3 cancelamento.py
```

A saída tem oito partes numeradas e é a mesma das seções 3 a 6 da aula. Ela sai igual toda vez: não tem sorteio.

```bash
python3 testar_meu_modelo.py
python3 meu_modelo.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError`. É o esperado. Quando as quatro funções estiverem certas, o teste mostra `ok: 11 testes passaram`. Se um teste falhar, a mensagem diz o que saiu e o que era esperado. O `meu_modelo.py` grava o `modelo.json` nessa pasta; rodar de novo grava por cima.

Se você rodar de outra pasta, os programas param com uma mensagem que diz de onde rodar. Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
