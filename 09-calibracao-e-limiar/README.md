# 09. Calibração e limiar

Os arquivos da aula «Calibração e limiar» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `modelo.py` | A preparação dos pedidos, o modelo de cancelamento com e sem peso, o acerto do par e a correção da escala. Os outros dois programas usam ele. |
| `calibracao.py` | As contas da aula: a ordem dos dois modelos, a curva de calibração, a decisão pelo corte do empate, a correção ajustada em abril e medida em maio e a tabela de custos. |
| `meu_corte.py` | Onde você escreve as funções `tabela_calibracao` e `custo` do exercício. Os preços da mensagem do Caio já estão nele. |
| `testar_meu_corte.py` | Os testes das suas duas funções, com seis pedidos inventados dentro do próprio teste. |
| `minhas-respostas.md` | Onde você escreve as contas à mão e as respostas do exercício. |

Os pedidos vêm de `../dados/loja/pedidos_export_completo.csv`, a base compartilhada do kit. Os programas tiram junho logo na leitura: junho é o teste e fica fechado.

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso.

```bash
python3 calibracao.py
```

A saída tem cinco partes numeradas e é a mesma das seções 3 a 7 da aula. Ela sai igual toda vez: não tem sorteio.

Para salvar o desenho da curva em `curva-calibracao.png`, rode `python3 calibracao.py --grafico`. Esse comando precisa do matplotlib: `python3 -m pip install matplotlib`. A aula foi rodada com o matplotlib 3.9.4 no Python 3.9.

```bash
python3 testar_meu_corte.py
python3 meu_corte.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 11 testes passaram`. Se um teste falhar, a mensagem diz o que saiu e o que era esperado.

Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
