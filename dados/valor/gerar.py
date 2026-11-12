"""Gera a base de valor dos pedidos da Loja Exemplo usada nas aulas de regressão.

Só biblioteca padrão (Python 3.9 ou mais novo). Semente fixa: rodar duas vezes dá o mesmo
arquivo, byte a byte.

    python3 gerar.py                 # escreve pedidos_valor.csv nesta pasta
    python3 gerar.py --saida /tmp/x  # escreve em outra pasta

Uma linha é um pedido entregue entre 2026-01-01 e 2026-06-30, com o que a loja sabe quando o
carrinho fecha e o valor pago, em reais.

Nas aulas, você ajusta e avalia primeiro e só depois lê este programa para comparar.
"""
import argparse
import csv
import datetime
import math
import pathlib
import random

SEMENTE = 2026
N = 1500
INICIO = datetime.date(2026, 1, 1)
FIM = datetime.date(2026, 6, 30)

# categoria, peso no sorteio, preço de um item (R$), peso de um item (kg)
CATEGORIAS = [("Eletro", 20, 160.0, 2.4), ("Casa", 24, 70.0, 1.6), ("Papelaria", 18, 18.0, 0.3),
              ("Esporte", 20, 110.0, 1.1), ("Beleza", 18, 45.0, 0.4)]
EFEITO_CUPOM = 25.0
FATOR_RECORRENTE = 1.35
CURVA_ITENS = 0.85
N_EMPRESAS = 12


def _sorteio(rng, opcoes):
    return rng.choices(opcoes, weights=[o[1] for o in opcoes])[0]


def gerar(saida):
    rng = random.Random(SEMENTE)
    saida.mkdir(parents=True, exist_ok=True)
    dias = (FIM - INICIO).days
    empresas = set(rng.sample(range(1, N + 1), N_EMPRESAS))
    linhas = []
    for i in range(1, N + 1):
        data = INICIO + datetime.timedelta(days=rng.randint(0, dias))
        nome, _, preco, peso_item = _sorteio(rng, CATEGORIAS)
        if i in empresas:
            itens = rng.randint(40, 80)
        else:
            itens = min(1 + int(rng.expovariate(1 / 1.6)), 12)
        cliente_novo = 1 if rng.random() < 0.35 else 0
        cupom = 1 if rng.random() < (0.55 if cliente_novo else 0.12) else 0
        canal = "app" if rng.random() < 0.55 else "site"
        visitas = max(0, int(rng.gauss(4 + 2 * (1 - cliente_novo), 2)))
        prazo = rng.randint(2, 15)
        hora = rng.randint(0, 23)
        peso = round(itens * peso_item * rng.uniform(0.8, 1.2), 2)

        base = preco * itens ** CURVA_ITENS
        if i in empresas:
            base *= 0.6
        if not cliente_novo:
            base *= FATOR_RECORRENTE
        valor = base * math.exp(rng.gauss(0, 0.18)) + EFEITO_CUPOM * cupom + 1.5 * (canal == "app")
        linhas.append({
            "id_pedido": i, "data_pedido": data.isoformat(), "categoria": nome, "itens": itens,
            "peso_kg": "%.2f" % peso, "cliente_novo": cliente_novo, "cupom": cupom, "canal": canal,
            "visitas_antes": visitas, "prazo_prometido_dias": prazo, "hora_pedido": hora,
            "dia_semana": data.isoweekday(), "valor": "%.2f" % valor,
        })
    colunas = list(linhas[0])
    with open(saida / "pedidos_valor.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colunas, lineterminator="\n")
        w.writeheader()
        w.writerows(linhas)
    return {"pedidos": len(linhas), "com_cupom": sum(l["cupom"] for l in linhas),
            "clientes_novos": sum(l["cliente_novo"] for l in linhas),
            "valor_total": "%.2f" % sum(float(l["valor"]) for l in linhas)}


def main():
    parser = argparse.ArgumentParser(description="Gera pedidos_valor.csv da Loja Exemplo.")
    parser.add_argument("--saida", default=str(pathlib.Path(__file__).resolve().parent),
                        help="pasta de saída (padrão: esta pasta)")
    resumo = gerar(pathlib.Path(parser.parse_args().saida))
    print(" ".join("%s=%s" % (k, v) for k, v in resumo.items()))


if __name__ == "__main__":
    main()
