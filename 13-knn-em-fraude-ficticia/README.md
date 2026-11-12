# 13. KNN em fraude

Os arquivos da aula «KNN em fraude» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `benchmark.py` | As contas da aula: as dobras com as fraudes repartidas, o custo de cada K e de cada corte do voto, a escolha, o tempo de uma previsão e o registro. Com `--abrir-teste`, mede junho uma vez. Também é o módulo que o exercício usa. |
| `minha_escolha.py` | Onde você escreve as funções `dobras_estratificadas` e `custo_da_regra` do exercício. Ele já roda a escolha com o preço do Diego, R$ 40. |
| `testar_minha_escolha.py` | Os testes das suas duas funções, com compras inventadas e com as dobras e os custos da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão, as anotações antes de abrir junho e a resposta do exercício. |

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso. Essa aula lê `dados/fraude/transacoes.csv`, na raiz do kit, e não usa o scikit-learn.

```bash
python3 benchmark.py
```

Ele leva uns dez segundos e grava o `benchmark.json` nessa pasta. As partes 1 a 4 e 6 saem iguais às da aula. Os tempos da parte 5 dependem do seu computador e mudam um pouco de uma rodada para outra. Rodando de novo, ele compara tudo com o `benchmark.json`, menos os tempos.

```bash
python3 benchmark.py --abrir-teste
```

Mede junho com a regra escolhida e grava o resultado no `benchmark.json`. Da segunda vez, ele recusa e sai com código 1. Para começar do zero, apague o `benchmark.json`.

```bash
python3 testar_minha_escolha.py
python3 minha_escolha.py
python3 minha_escolha.py --abrir-teste
```

Antes de você escrever as funções, o teste para com `NotImplementedError: escreva custo_da_regra` e o `minha_escolha.py` para com `NotImplementedError: escreva dobras_estratificadas`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 13 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual é o certo. O `minha_escolha.py` grava o `meu-benchmark.json`, separado do registro da aula.

Se aparecer `não achei`, você rodou fora da pasta da aula ou a pasta `dados/` não está na raiz do kit. Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
