"""Gera as reservas da Agência Exemplo usadas no case de viagens.

Só biblioteca padrão (Python 3.9 ou mais novo). Semente fixa: rodar duas vezes dá o mesmo
arquivo, byte a byte.

    python3 gerar.py                 # escreve reservas.csv nesta pasta
    python3 gerar.py --saida /tmp/x  # escreve em outra pasta

Uma linha é uma reserva de pacote feita entre 2025-01-01 e 2026-03-31, com viagem até
2026-09-30. O rótulo `cancelou` é 1 quando a reserva foi cancelada antes da viagem. A base foi
extraída em 2026-10-01, depois da última viagem, então todo rótulo já existe.

`valor_reembolsado` é o que o financeiro devolveu ao cliente.

Nas aulas, você treina e avalia primeiro e só depois lê este programa para comparar.
"""
import argparse
import csv
import datetime
import math
import pathlib
import random

SEMENTE = 3107
N = 3600
INICIO = datetime.date(2025, 1, 1)
FIM = datetime.date(2026, 3, 31)
INICIO_FLEX_PROMO = datetime.date(2026, 1, 1)  # campanha de tarifa flexível
DESTINOS = [("praia", 38, 2400), ("serra", 20, 1900), ("cidade", 27, 1500), ("exterior", 15, 6800)]
CANAIS = [("site", 45, 0.0), ("app", 30, -0.1), ("telefone", 10, -0.3), ("parceiro", 15, 0.45)]
PAGAMENTOS = [("pix", 40, -0.2), ("cartao_parcelado", 45, 0.0), ("boleto", 15, 0.5)]


def _sorteio(rng, opcoes):
    return rng.choices(opcoes, weights=[o[1] for o in opcoes])[0]


def gerar(saida):
    rng = random.Random(SEMENTE)
    saida.mkdir(parents=True, exist_ok=True)
    dias = (FIM - INICIO).days
    linhas = []
    for _ in range(N):
        reserva = INICIO + datetime.timedelta(days=rng.randint(0, dias))
        antecedencia = max(1, min(180, int(rng.expovariate(1 / 45)) + 1))
        viagem = reserva + datetime.timedelta(days=antecedencia)
        destino, _, preco = _sorteio(rng, DESTINOS)
        canal, _, ef_canal = _sorteio(rng, CANAIS)
        pagamento, _, ef_pag = _sorteio(rng, PAGAMENTOS)
        noites = rng.randint(2, 10)
        adultos = rng.choices([1, 2, 3, 4], weights=[20, 55, 15, 10])[0]
        criancas = rng.choices([0, 1, 2, 3], weights=[60, 22, 14, 4])[0]
        anteriores = 0 if rng.random() < 0.55 else min(int(rng.expovariate(1 / 2.0)) + 1, 12)
        p_flex = 0.45 if reserva >= INICIO_FLEX_PROMO else 0.22
        flexivel = 1 if rng.random() < p_flex else 0
        valor = preco * (adultos + 0.5 * criancas) * noites / 5 * math.exp(rng.gauss(0, 0.2))

        x = math.log(antecedencia)
        logito = (-2.6 + 0.9 * (x - 3.2) + 0.35 * (x - 3.2) ** 2
                  + 1.1 * flexivel + 0.25 * flexivel * (x - 3.2)
                  + ef_canal + ef_pag - 0.12 * min(anteriores, 6)
                  + 0.25 * (criancas > 0) + 0.3 * (destino == "exterior")
                  + rng.gauss(0, 0.5))
        cancelou = 1 if rng.random() < 1 / (1 + math.exp(-logito)) else 0
        if cancelou:
            reembolso = valor * (0.9 if flexivel else rng.choice([0.0, 0.0, 0.3, 0.5]))
        else:
            reembolso = 0.0
        linhas.append({
            "data_reserva": reserva.isoformat(), "data_viagem": viagem.isoformat(),
            "antecedencia_dias": antecedencia, "destino": destino, "noites": noites,
            "adultos": adultos, "criancas": criancas, "valor_total": "%.2f" % valor,
            "canal": canal, "pagamento": pagamento, "tarifa_flexivel": flexivel,
            "reservas_anteriores": anteriores, "valor_reembolsado": "%.2f" % reembolso,
            "cancelou": cancelou,
        })
    linhas.sort(key=lambda l: l["data_reserva"])
    for i, l in enumerate(linhas, start=1):
        l["id_reserva"] = i
    colunas = ["id_reserva"] + [c for c in linhas[0] if c != "id_reserva"]
    with open(saida / "reservas.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colunas, lineterminator="\n")
        w.writeheader()
        w.writerows(linhas)
    return {"reservas": len(linhas), "canceladas": sum(l["cancelou"] for l in linhas),
            "flexiveis": sum(l["tarifa_flexivel"] for l in linhas)}


def main():
    parser = argparse.ArgumentParser(description="Gera reservas.csv da Agência Exemplo.")
    parser.add_argument("--saida", default=str(pathlib.Path(__file__).resolve().parent),
                        help="pasta de saída (padrão: esta pasta)")
    resumo = gerar(pathlib.Path(parser.parse_args().saida))
    print(" ".join("%s=%s" % (k, v) for k, v in resumo.items()))


if __name__ == "__main__":
    main()
