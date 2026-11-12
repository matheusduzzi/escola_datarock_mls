"""Exercício da aula «SVM na Loja Exemplo»: o valor em centavos e o teto dobrado.

Escreva as duas funções abaixo. Depois rode, de dentro da pasta da aula:

    python3 testar_meu_svm.py
    python3 meu_svm.py

Antes de você escrever as funções, os dois param com NotImplementedError.
"""
import numpy as np
import pandas as pd

from svm_loja import (CATEGORIAS, NUMERICAS, NaoParou, acerto_do_par, br, dobras, gamma_scale,
                      k_linear, k_rbf, kernel_da_grade, ler, nota_logistica, pct, sinal, treinar,
                      treinar_logistica, decisao)


def padronizar(treino, outro):
    """Padroniza as colunas de `treino` e de `outro` pela média e pelo desvio de `treino`.

    Recebe dois DataFrames do pandas com as mesmas colunas, todas numéricas. Retorna dois DataFrames
    novos: (treino padronizado, outro padronizado). O desvio é o da população (ddof=0).
    Para com ValueError se as colunas forem diferentes, se faltar valor em qualquer um dos dois
    ou se uma coluna tiver desvio zero no treino. Não muda o que recebeu.
    """
    raise NotImplementedError("escreva padronizar(treino, outro)")


def no_teto(f, y, quantos):
    """Quantos cancelados ficam entre os `quantos` pedidos de f mais alto.

    `f` e `y` são Series do pandas com o mesmo índice (o número do pedido). `y` vale 0 ou 1.
    Em empate de f, vem antes o pedido de número menor. Para com ValueError se os índices forem
    diferentes, se `quantos` for menor que 1 ou maior que o número de pedidos, se faltar f ou se
    `y` tiver algo além de 0 e 1. Não muda o que recebeu.
    """
    raise NotImplementedError("escreva no_teto(f, y, quantos)")


# ---------- daqui para baixo já está pronto ----------

def matriz(treino, outro, em_centavos=False, padronizada=True):
    """As 14 colunas da aula, com o valor em reais ou em centavos. As numéricas passam pelo seu padronizar."""
    num_tr = treino[NUMERICAS].astype(float).copy()
    num_ou = outro[NUMERICAS].astype(float).copy()
    if em_centavos:
        num_tr["valor"] = (num_tr["valor"] * 100).round()
        num_ou["valor"] = (num_ou["valor"] * 100).round()
    if padronizada:
        num_tr, num_ou = padronizar(num_tr, num_ou)
    partes_tr, partes_ou = [num_tr.to_numpy()], [num_ou.to_numpy()]
    for c in CATEGORIAS:
        for v in sorted(treino[c].unique()):
            partes_tr.append((treino[c] == v).to_numpy(dtype=float)[:, None])
            partes_ou.append((outro[c] == v).to_numpy(dtype=float)[:, None])
    return np.hstack(partes_tr), np.hstack(partes_ou)


def parte_a(ds):
    nome, tr, me = ds[2]
    y = sinal(tr["cancelado"])
    print(f"A. O valor em centavos: a dobra de {nome}, treino de janeiro a março")
    fs = {}
    for padronizada in (True, False):
        for em_centavos in (False, True):
            x, xm = matriz(tr, me, em_centavos, padronizada)
            unidade = "centavos" if em_centavos else "reais"
            jeito = "padronizado" if padronizada else "sem padronizar"
            try:
                mo = treinar(x, y, k_linear, 0.01)
                f = decisao(mo, xm)
                fs[(padronizada, em_centavos)] = f
                print(f"  reta, C = 0,01, valor em {unidade}, {jeito}: {mo['passos']} passos, "
                      f"acerto do par {pct(acerto_do_par(f, me['cancelado']))}, f de {br(f.min(), 4)} a {br(f.max(), 4)}")
            except NaoParou:
                print(f"  reta, C = 0,01, valor em {unidade}, {jeito}: não parou")
            g = gamma_scale(x)
            mo = treinar(x, y, k_rbf(g), 1.0)
            print(f"  RBF, gamma 'scale' = {br(g, 10) if g < 1e-4 else br(g, 4)}, C = 1, valor em {unidade}, {jeito}: "
                  f"acerto do par {pct(acerto_do_par(decisao(mo, xm), me['cancelado']))}")
    dif = float(np.abs(fs[(True, False)] - fs[(True, True)]).max())
    print(f"  padronizado, maior diferença de f entre reais e centavos nos {len(me)} pedidos: "
          f"{'abaixo de 0,000001' if dif < 1e-6 else br(dif, 6)}")
    print()


def parte_b(ds):
    print("B. O teto dobrado: cancelados entre os pedidos de f mais alto, em cada dobra")
    print(f"  {'mês':<11}{'pedidos':>8}{'cancelados':>12}  {'1 em 4':>6} {'log':>5}{'reta':>6}{'RBF':>5}"
          f"  {'1 em 2':>6} {'log':>5}{'reta':>6}{'RBF':>5}")
    soma = np.zeros(8, dtype=int)
    for nome, tr, me in ds:
        x, xm = matriz(tr, me)
        y = sinal(tr["cancelado"])
        ym = pd.Series(me["cancelado"].to_numpy(), index=me["id_pedido"].to_numpy())
        f_log = pd.Series(nota_logistica(treinar_logistica(x, tr["cancelado"]), xm), index=ym.index)
        f_ret = pd.Series(decisao(treinar(x, y, k_linear, 0.01), xm), index=ym.index)
        k, _ = kernel_da_grade("RBF", 0.25, x)
        f_rbf = pd.Series(decisao(treinar(x, y, k, 0.1), xm), index=ym.index)
        linha = []
        for divisor in (4, 2):
            q = len(me) // divisor
            linha += [q] + [no_teto(f_, ym, q) for f_ in (f_log, f_ret, f_rbf)]
        soma += np.array(linha)
        print(f"  {nome:<11}{len(me):>8}{int(ym.sum()):>12}  {linha[0]:>6} {linha[1]:>5}{linha[2]:>6}{linha[3]:>5}"
              f"  {linha[4]:>6} {linha[5]:>5}{linha[6]:>6}{linha[7]:>5}")
    total = sum(int(me["cancelado"].sum()) for _, _, me in ds)
    print(f"  {'soma':<11}{sum(len(me) for _, _, me in ds):>8}{total:>12}  {soma[0]:>6} {soma[1]:>5}{soma[2]:>6}"
          f"{soma[3]:>5}  {soma[4]:>6} {soma[5]:>5}{soma[6]:>6}{soma[7]:>5}")
    print("  log: logística; reta: reta do SVM, C = 0,01; RBF: gamma × 0,25, C = 0,1")


def main():
    ds = dobras(ler())
    parte_a(ds)
    parte_b(ds)


if __name__ == "__main__":
    main()
