# Mensagens ao atendimento da Loja Exemplo

A Loja Exemplo não existe. As mensagens são montadas por `gerar.py`, que só usa a biblioteca padrão do Python, a partir de frases escritas para o curso. Nenhuma mensagem vem de cliente real, e nenhuma tem nome, telefone, e-mail ou documento.

## Arquivo

`mensagens.csv` tem 1800 linhas, em ordem de data. Uma linha é a primeira mensagem de um atendimento aberto entre 2026-01-01 e 2026-06-30.

| Coluna | O que é |
|---|---|
| `id_mensagem` | número da mensagem, na ordem do tempo |
| `data` | dia em que o atendimento abriu |
| `canal` | `chat`, `app` ou `email` |
| `texto` | o que o cliente escreveu, do jeito que chegou: com ou sem acento, às vezes em maiúsculas |
| `assunto` | o que a pessoa do atendimento marcou ao fechar: `atraso`, `troca`, `pagamento` ou `outro` |

## Como gerar de novo

```bash
python3 gerar.py
```

A saída é `mensagens=1800 atraso=669 troca=495 pagamento=362 outro=274`. O arquivo sai igual toda vez. Nas aulas, você treina e avalia primeiro e só depois abre o gerador para comparar.

Não edite o CSV à mão.
