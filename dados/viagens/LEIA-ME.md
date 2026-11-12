# Reservas da Agência Exemplo

A Agência Exemplo é uma agência de viagens inventada para o case final. As reservas são números sorteados por `gerar.py`, que só usa a biblioteca padrão do Python. Nenhum dado vem de cliente, pessoa ou empresa real.

## Arquivo

`reservas.csv` tem 3600 linhas, em ordem de data de reserva. Uma linha é uma reserva de pacote feita entre 2025-01-01 e 2026-03-31. A base saiu do sistema em 2026-10-01, depois da última viagem.

| Coluna | O que é |
|---|---|
| `id_reserva` | número da reserva, na ordem do tempo |
| `data_reserva` | dia em que a reserva foi feita |
| `data_viagem` | dia da ida |
| `antecedencia_dias` | dias entre a reserva e a viagem |
| `destino` | `praia`, `serra`, `cidade` ou `exterior` |
| `noites` | quantas noites o pacote tem |
| `adultos`, `criancas` | quantas pessoas viajam |
| `valor_total` | preço do pacote, em reais |
| `canal` | `site`, `app`, `telefone` ou `parceiro` |
| `pagamento` | `pix`, `cartao_parcelado` ou `boleto` |
| `tarifa_flexivel` | 1 se o cliente pagou pela tarifa que permite cancelar com reembolso maior |
| `reservas_anteriores` | quantas reservas o cliente já tinha feito na agência |
| `valor_reembolsado` | quanto o financeiro devolveu ao cliente |
| `cancelou` | 1 quando a reserva foi cancelada antes da viagem |

## Como gerar de novo

```bash
python3 gerar.py
```

A saída é `reservas=3600 canceladas=736 flexiveis=993`. O arquivo sai igual toda vez. No case, você escolhe e defende o modelo primeiro e só depois abre o gerador para comparar.

Não edite o CSV à mão.
