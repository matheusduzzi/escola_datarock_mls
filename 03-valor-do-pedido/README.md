# 03. Valor do pedido

Os arquivos da aula «Valor do pedido» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `valor_pedido.py` | As contas da aula com os 1500 pedidos: as médias com e sem cupom, a regressão com uma e com duas colunas, o modelo da loja, os resíduos por faixa de itens, o modelo sem os 12 pedidos de empresa, os resíduos por faixa e categoria, a inclinação dos itens por categoria, o caminho do cupom e o intervalo dos coeficientes. |
| `regressao.py` | As funções que a aula e o exercício usam: `carregar`, `ajustar`, `residuos_por`, `intervalo` e `br`. O começo do arquivo explica os nomes de coluna que você pode passar. |
| `minha_analise.py` | Onde você escreve o código do exercício. Ele já lê os pedidos e separa os 1488 de varejo. |
| `minhas-respostas.md` | Onde você escreve as respostas do exercício. |

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, como no README do kit do curso. Essa aula lê `../dados/valor/pedidos_valor.csv`.

```bash
python3 valor_pedido.py
```

A saída tem nove partes e é a mesma da aula. Ela sai igual toda vez, inclusive os 500 sorteios da parte 9.

```bash
python3 minha_analise.py
```

Antes de você escrever o exercício, a saída é `1488 pedidos de varejo`.

Para os trechos da prática guiada, abra o `python3` nessa pasta e cole o código. O `regressao.py` precisa estar na pasta de onde você abriu o `python3`.

Se aparecer `erro: não achei ...pedidos_valor.csv`, você rodou de fora da pasta da aula ou a pasta `dados/` não está ao lado dela. Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.
