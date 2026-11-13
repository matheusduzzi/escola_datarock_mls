# 04. MAE, RMSE e R²

Os arquivos da aula «MAE, RMSE e R²» do curso ML supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `medir_erro.py` | As contas da aula: o R² medido nos próprios pedidos do ajuste, os mesmos erros em três réguas, a divisão por data, quatro modelos medidos em maio e junho, o mesmo com os pedidos de empresa, o erro por tipo de cliente e o R² ajustado com colunas sorteadas. |
| `metricas.py` | As funções que a aula e o exercício usam: `mae`, `rmse`, `r2`, `r2_ajustado`, `erro_medio`, `medir`, `dividir_por_data`, `prever` e `treinar_e_medir`. Cada métrica é uma linha de NumPy. |
| `regressao.py` | O mesmo arquivo da aula «Valor do pedido», com `carregar`, `ajustar` e `br`. Ele está aqui para a pasta rodar sozinha. |
| `minha_escolha.py` | Onde você escreve o código do exercício. Ele já separa treino e teste, ajusta o modelo e calcula a previsão de cada pedido do teste. |
| `minhas-respostas.md` | Onde você escreve as respostas do exercício. |

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, como no README do kit do curso. Essa aula lê `../dados/valor/pedidos_valor.csv`.

```bash
python3 medir_erro.py
```

A saída tem sete partes e é a mesma da aula. Ela sai igual toda vez, inclusive as colunas sorteadas da parte 7.

```bash
python3 minha_escolha.py
```

Antes de você escrever o exercício, a saída é `963 pedidos de treino, 525 de teste`.

Para os trechos da prática guiada, abra o `python3` nessa pasta e cole o código. O `metricas.py` e o `regressao.py` precisam estar na pasta de onde você abriu o `python3`.

Se aparecer `erro: não achei ...pedidos_valor.csv`, você rodou de fora da pasta da aula ou a pasta `dados/` não está ao lado dela. Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

Se você tem o scikit-learn, pode refazer as três primeiras métricas com `mean_absolute_error`, `root_mean_squared_error` (desde a versão 1.4) e `r2_score`, do `sklearn.metrics`. O R² ajustado ele não tem; a fórmula está no `metricas.py`.
