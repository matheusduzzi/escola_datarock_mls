# 11. Escolher regularização

Os arquivos da aula «Escolher regularização» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `experimento.py` | As contas da aula: os pedidos de janeiro a maio, as 21 colunas, as cinco dobras, o erro de cada alfa do Ridge, a escolha, o registro `experimento.json` e a abertura de junho, uma vez. |
| `minha_escolha.py` | Onde você escreve as funções `dividir_em_dobras` e `erro_nas_dobras` do exercício. A grade do Lasso já está nele. |
| `testar_minha_escolha.py` | Os testes das suas duas funções, com seis pedidos inventados dentro do próprio teste. |
| `minhas-respostas.md` | Onde você escreve as contas à mão, as anotações de antes de abrir junho e a resposta para o Caio. |

Os pedidos vêm de `../dados/valor/pedidos_valor.csv`, a base compartilhada do kit. Os programas leem só janeiro a maio. Junho só é lido com `--abrir-teste`.

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso.

```bash
python3 experimento.py
```

A saída tem quatro partes numeradas e é a mesma das seções 3 a 6 da aula. Na primeira vez, ela grava o `experimento.json` nessa pasta. Nas outras, ela refaz a conta e compara com o registro: igual, diz igual; diferente, lista o que mudou e sai com código 1.

```bash
python3 experimento.py --sorteios
python3 experimento.py --abrir-teste
```

O `--sorteios` repete a escolha em 20 sorteios das dobras e não grava nada. O `--abrir-teste` refaz a escolha, exige que ela bata com o registro e só então mede junho. Ele abre uma vez: na segunda, mostra o resultado da primeira e sai com código 1. É o esperado.

Para o exercício:

```bash
python3 testar_minha_escolha.py
python3 minha_escolha.py
python3 minha_escolha.py --abrir-teste
```

Antes de você escrever as funções, os dois primeiros param com `NotImplementedError`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 10 testes passaram`. Se um teste falhar, a mensagem diz o que saiu e o que era esperado. O `minha_escolha.py` grava o seu registro em `meu-experimento.json` e leva alguns segundos, porque o Lasso ajusta coluna por coluna.

Quer recomeçar do zero? Apague o `experimento.json` ou o `meu-experimento.json`. Isso também apaga o que dizia que junho já foi aberto, então anote nas suas respostas que você recomeçou.

Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
