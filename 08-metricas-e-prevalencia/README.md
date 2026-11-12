# 08. Métricas e prevalência

Os arquivos da aula «Métricas e prevalência» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `modelo.py` | O modelo de cancelamento da aula 07, o modelo de fraude e as medidas da aula: precisão, revocação, F1, taxa de falso positivo, acerto do par, KS e precisão média. Os outros dois programas usam ele. |
| `metricas.py` | As contas da aula: as duas bases, o cancelamento em quatro cortes e na lista inteira, a fraude com a regra do valor e com o modelo, a lista de revisão e a mesma regra em outra mistura. |
| `comparar.py` | O exercício: o modelo da aula contra o modelo da Bruna, com a coluna da madrugada. |
| `minhas-respostas.md` | Onde você escreve as contas à mão e a resposta para a Bruna. |

Os pedidos vêm de `../dados/loja/pedidos_export_completo.csv` e as compras com cartão, de `../dados/fraude/transacoes.csv`, as bases compartilhadas do kit. Os programas tiram junho logo na leitura: junho é o teste e fica fechado.

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso.

```bash
python3 metricas.py
```

A saída tem seis partes numeradas e é a mesma das seções 3 a 7 da aula. Ela sai igual toda vez: não tem sorteio.

Para ver os pontos das curvas ROC e PR da fraude, um por linha, rode `python3 metricas.py --curvas`. Cada linha tem o modelo, o corte, a revocação, a taxa de falso positivo e a precisão. Dá para abrir numa planilha e desenhar as duas curvas.

```bash
python3 comparar.py
```

A saída tem duas partes: as medidas da lista inteira e as listas de revisão com 27, 54 e 100 compras.

Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
