"""Gera o entregas.csv da aula «SVM e kernels».

Só biblioteca padrão (Python 3.9 ou mais novo). Rodar duas vezes dá o mesmo arquivo, byte a byte.

    python3 gerar_entregas.py

Cada linha é um pedido entregue pelos motoboys que saem do centro de distribuição da Loja
Exemplo, em março ou abril de 2026. leste_km e norte_km dizem onde fica o endereço de entrega,
em km a partir do centro de distribuição (negativo é oeste ou sul). cancelado é 1 quando o
cliente cancelou o pedido antes de receber.

Nas aulas, você olha os pedidos primeiro e só depois abre este programa para comparar.
"""
import csv
import datetime
import math
import pathlib
import random

PONTO_DE_PARTIDA = 16
RAIO_MAXIMO_KM = 12.0
RAIO_DO_ATRASO_KM = 9.0
LARGURA_KM = 0.6
MESES = (("2026-03", 31, 80), ("2026-04", 30, 40))


def gerar(saida):
    rng = random.Random(PONTO_DE_PARTIDA)
    linhas, n = [], 0
    for mes, dias, quantos in MESES:
        for _ in range(quantos):
            n += 1
            raio = RAIO_MAXIMO_KM * math.sqrt(rng.random())
            angulo = rng.uniform(0, 2 * math.pi)
            leste, norte = round(raio * math.cos(angulo), 1), round(raio * math.sin(angulo), 1)
            dist = math.hypot(leste, norte)
            chance = 1 / (1 + math.exp(-(dist - RAIO_DO_ATRASO_KM) / LARGURA_KM))
            linhas.append({"id_pedido": 5000 + n,
                           "data_pedido": f"{mes}-{rng.randint(1, dias):02d}",
                           "leste_km": f"{leste:.1f}", "norte_km": f"{norte:.1f}",
                           "cancelado": 1 if rng.random() < chance else 0})
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(linhas)
    return linhas


if __name__ == "__main__":
    destino = pathlib.Path(__file__).resolve().parent / "entregas.csv"
    linhas = gerar(destino)
    marco = [l for l in linhas if l["data_pedido"].startswith("2026-03")]
    print(f"pedidos={len(linhas)} marco={len(marco)} abril={len(linhas) - len(marco)} "
          f"cancelados={sum(l['cancelado'] for l in linhas)}")
