# Valor dos pedidos da Loja Exemplo

A Loja Exemplo não existe. Os pedidos são números sorteados por `gerar.py`, que só usa a biblioteca padrão do Python. Nenhum dado vem de cliente, pessoa ou empresa real.

## Arquivo

`pedidos_valor.csv` tem 1500 linhas. Uma linha é um pedido entregue entre 2026-01-01 e 2026-06-30.

| Coluna | O que é |
|---|---|
| `id_pedido` | número do pedido |
| `data_pedido` | dia em que o carrinho fechou |
| `categoria` | Eletro, Casa, Papelaria, Esporte ou Beleza |
| `itens` | quantos itens o pedido tem |
| `peso_kg` | peso total, em quilos |
| `cliente_novo` | 1 se é a primeira compra do cliente |
| `cupom` | 1 se o pedido usou cupom |
| `canal` | `app` ou `site` |
| `visitas_antes` | quantas vezes o cliente entrou na loja nos 30 dias antes do pedido |
| `prazo_prometido_dias` | prazo de entrega mostrado no carrinho |
| `hora_pedido` | hora do dia, de 0 a 23 |
| `dia_semana` | 1 é segunda-feira e 7 é domingo |
| `valor` | quanto o cliente pagou, em reais; é o que as aulas de regressão preveem |

## Como gerar de novo

```bash
python3 gerar.py
```

A saída é `pedidos=1500 com_cupom=436 clientes_novos=564 valor_total=320875.21`. O arquivo sai igual toda vez. Nas aulas, você ajusta e avalia primeiro e só depois abre o gerador para comparar.

Não edite o CSV à mão. Se uma aula precisar de outro caso, ela traz os arquivos dela na própria pasta.
