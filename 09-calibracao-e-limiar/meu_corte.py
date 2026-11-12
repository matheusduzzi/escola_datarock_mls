"""Exercício da aula «Calibração e limiar»: a mensagem antes do despacho.

Escreva as duas funções marcadas com NotImplementedError. Depois rode, de dentro da pasta da aula:

    python3 testar_meu_corte.py
    python3 meu_corte.py

Enquanto as funções não existem, os dois comandos param com NotImplementedError.
"""
import numpy as np

import modelo as m

BASE = "../dados/loja/pedidos_export_completo.csv"

# A proposta do Caio: uma mensagem antes do despacho, sem teto de envios.
PRECO_MENSAGEM = 1.0        # reais por mensagem enviada
PRECO_CANCELAMENTO = 15.0   # reais por cancelamento que acontece
EVITA = 0.5                 # parte dos cancelamentos que a mensagem evita, pela estimativa do Caio
LIMITES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
CORTES = [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]


def tabela_calibracao(nota, y, limites):
    """Uma linha por faixa de nota, só as faixas com pedido.

    Cada linha é (início, fim, pedidos, nota média, parte que cancelou). Uma nota igual a um
    limite entra na faixa de cima; a nota 1 entra na última faixa.
    Pare com ValueError se nota e y tiverem tamanhos diferentes, se alguma nota estiver fora
    de 0 a 1, ou se os limites não começarem em 0, não terminarem em 1 ou não crescerem.
    """
    raise NotImplementedError("escreva tabela_calibracao(nota, y, limites)")


def custo(nota, y, corte, preco_contato, preco_cancelamento, evita=1.0):
    """Retorna (contatos, contatos com quem ia cancelar, custo em reais) da regra
    "contata quem tem nota maior ou igual ao corte".

    O contato custa preco_contato. Um cancelamento sem contato custa preco_cancelamento. Um
    cancelamento com contato custa (1 - evita) × preco_cancelamento: a parte que o contato não evita.
    Pare com ValueError se nota e y tiverem tamanhos diferentes ou se evita não estiver
    entre 0 (fora) e 1 (dentro).
    """
    raise NotImplementedError("escreva custo(nota, y, corte, preco_contato, preco_cancelamento, evita)")


def main():
    d = m.preparar(m.ler(BASE))
    x_tr, y_tr, _ = d["treino"]
    x_va = np.r_[d["abril"][0], d["maio"][0]]
    y_va = np.r_[d["abril"][1], d["maio"][1]]
    nota = m.nota(m.treinar(x_tr, y_tr), x_va)       # o modelo sem peso da aula

    print(f"modelo sem peso, abril e maio: {len(y_va)} pedidos, {y_va.sum()} cancelados")
    print("\ncurva de calibração")
    print("faixa              pedidos  nota média  cancelaram")
    for inicio, fim, n, media, real in tabela_calibracao(nota, y_va, LIMITES):
        print(f"{m.num(inicio, 2)} a {m.num(fim, 2)}{n:>14}{m.pct(media):>12}{m.pct(real):>12}")

    print(f"\ntabela de custos: mensagem R$ {m.num(PRECO_MENSAGEM, 2)}, cancelamento R$ "
          f"{m.num(PRECO_CANCELAMENTO, 2)}, a mensagem evita {m.pct(EVITA, 0)}")
    print("corte       mensagens  com quem ia cancelar       custo")
    contatos, certos, total = custo(nota, y_va, 2.0, PRECO_MENSAGEM, PRECO_CANCELAMENTO, EVITA)
    print(f"{'nenhuma':<10}{contatos:>11}{certos:>22}  R$ {m.num(total, 2):>9}")
    for corte in CORTES:
        contatos, certos, total = custo(nota, y_va, corte, PRECO_MENSAGEM, PRECO_CANCELAMENTO, EVITA)
        print(f"{m.num(corte, 3):<10}{contatos:>11}{certos:>22}  R$ {m.num(total, 2):>9}")


if __name__ == "__main__":
    main()
