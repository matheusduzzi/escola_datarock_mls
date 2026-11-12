"""Testes das funções treino_da_dobra e economia_da_lista do minha_comparacao.py.

Rode de dentro da pasta da aula:  python3 testar_minha_comparacao.py
Os testes usam casos pequenos feitos à mão e os números das partes 1 e 4 da aula. A leitura das
partes A, B e C do minha_comparacao.py continua com você.
"""
import inspect

import numpy as np
import pandas as pd

import minha_comparacao
from minha_comparacao import economia_da_lista, treino_da_dobra
from case_viagens import TRIMESTRES, ler, nota_do_candidato


def _erro(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def _pequena():
    return pd.DataFrame({
        "id_reserva": [1, 2, 3, 4, 5],
        "data_reserva": pd.to_datetime(["2025-05-02", "2025-05-10", "2025-06-01", "2025-06-20", "2025-07-01"]),
        "data_viagem": pd.to_datetime(["2025-05-20", "2025-07-01", "2025-06-30", "2025-08-15", "2025-07-05"]),
        "cancelou": [0, 1, 1, 0, 0],
    })


# ---------- treino_da_dobra ----------

def testar_so_entra_quem_tem_desfecho():
    tr = treino_da_dobra(_pequena(), "2025-07-01")
    assert list(tr["id_reserva"]) == [1, 3], (
        "no dia 1º de julho, só as reservas 1 e 3 já viajaram; a 2 viaja no próprio dia, a 4 depois "
        f"e a 5 foi feita no dia; veio {list(tr['id_reserva'])}")


def testar_viagem_no_dia_fica_fora():
    tr = treino_da_dobra(_pequena(), "2025-07-01")
    assert 2 not in set(tr["id_reserva"]), "a reserva 2 viaja em 1º de julho: no começo do dia, o desfecho dela ainda não existe"


def testar_texto_ou_timestamp():
    a = treino_da_dobra(_pequena(), "2025-07-01")
    b = treino_da_dobra(_pequena(), pd.Timestamp("2025-07-01"))
    assert a.equals(b), "'2025-07-01' e pd.Timestamp('2025-07-01') precisam dar o mesmo treino"


def testar_mantem_a_ordem_da_base():
    base = _pequena().iloc[::-1]
    tr = treino_da_dobra(base, "2025-07-01")
    assert list(tr["id_reserva"]) == [3, 1], f"a saída segue a ordem de base; veio {list(tr['id_reserva'])}"


def testar_falta_coluna():
    assert _erro(treino_da_dobra, _pequena().drop(columns=["data_viagem"]), "2025-07-01"), \
        "sem a coluna data_viagem, não dá para saber quem tem desfecho: pare com ValueError"
    assert _erro(treino_da_dobra, _pequena().drop(columns=["data_reserva"]), "2025-07-01"), \
        "sem a coluna data_reserva: pare com ValueError"


def testar_falta_data():
    base = _pequena()
    base.loc[3, "data_viagem"] = pd.NaT
    assert _erro(treino_da_dobra, base, "2025-07-01"), \
        "a reserva 4 está sem data de viagem; mesmo ficando fora pelo início, pare com ValueError"


def testar_viagem_antes_da_reserva():
    base = _pequena()
    base.loc[0, "data_viagem"] = pd.Timestamp("2025-04-30")
    assert _erro(treino_da_dobra, base, "2025-07-01"), "viagem antes da própria reserva: pare com ValueError"


def testar_nao_muda_o_que_recebeu():
    base = _pequena()
    for c in ("data_reserva", "data_viagem"):
        base[c] = base[c].dt.strftime("%Y-%m-%d")        # datas em texto, como vêm de um CSV sem parse_dates
    copia = base.copy()
    tr = treino_da_dobra(base, "2025-07-01")
    assert list(tr["id_reserva"]) == [1, 3], f"com as datas em texto, o treino é o mesmo: 1 e 3; veio {list(tr['id_reserva'])}"
    tr.loc[tr.index[0], "cancelou"] = 9
    assert base.equals(copia), "treino_da_dobra não pode mudar a base, nem por meio do que retorna (use .copy())"


def testar_numeros_da_parte_1():
    base = ler()
    tamanhos = [len(treino_da_dobra(base, ini)) for _, ini, _ in TRIMESTRES]
    assert tamanhos == [439, 1106, 1828], f"a parte 1 da aula tem 439, 1106 e 1828 reservas de treino; veio {tamanhos}"


# ---------- economia_da_lista ----------

NOTA = [0.9, 0.2, 0.7, 0.4, 0.1]
Y = [1, 0, 0, 1, 0]
IDS = [10, 11, 12, 13, 14]


def testar_conta_simples():
    pegas, e = economia_da_lista(NOTA, Y, IDS, 3, 20.0, 100.0)
    assert (pegas, e) == (2, 140.0), (
        "as 3 maiores notas são as reservas 10, 12 e 13; canceladas 10 e 13: 2 × 100 − 3 × 20 = 140; "
        f"veio {(pegas, e)}")


def testar_preco_e_valor_entram_na_conta():
    _, e = economia_da_lista(NOTA, Y, IDS, 3, 35.0, 100.0)
    assert e == 95.0, f"com a ligação a R$ 35: 2 × 100 − 3 × 35 = 95; veio {e}"


def testar_empate_pelo_numero_da_reserva():
    pegas, _ = economia_da_lista([0.5, 0.5, 0.5], [1, 0, 0], [30, 20, 10], 1, 20.0, 100.0)
    assert pegas == 0, ("três notas iguais: vem antes a reserva de número menor, a 10, que não cancelou; "
                        f"veio {pegas} cancelada na lista (desempate pela posição?)")


def testar_lista_inteira():
    pegas, e = economia_da_lista(NOTA, Y, IDS, 5, 20.0, 100.0)
    assert (pegas, e) == (2, 100.0), f"ligando para as 5: 2 × 100 − 5 × 20 = 100; veio {(pegas, e)}"


def testar_tamanhos_diferentes():
    assert _erro(economia_da_lista, NOTA, Y[:4], IDS, 2, 20.0, 100.0), "y com 4 e nota com 5: pare com ValueError"
    assert _erro(economia_da_lista, NOTA, Y, IDS[:4], 2, 20.0, 100.0), "ids com 4 e nota com 5: pare com ValueError"


def testar_quantos_fora_da_faixa():
    assert _erro(economia_da_lista, NOTA, Y, IDS, 0, 20.0, 100.0), "quantos = 0: pare com ValueError"
    assert _erro(economia_da_lista, NOTA, Y, IDS, 6, 20.0, 100.0), "quantos = 6 com 5 reservas: pare com ValueError"


def testar_nota_faltando():
    assert _erro(economia_da_lista, [0.9, float("nan"), 0.7, 0.4, 0.1], Y, IDS, 2, 20.0, 100.0), \
        "nota faltando numa reserva: pare com ValueError"


def testar_y_so_zero_e_um():
    assert _erro(economia_da_lista, NOTA, [1, 0, 2, 1, 0], IDS, 2, 20.0, 100.0), "y com 2: pare com ValueError"


def testar_ids_repetidos():
    assert _erro(economia_da_lista, NOTA, Y, [10, 11, 11, 13, 14], 2, 20.0, 100.0), \
        "a reserva 11 aparece duas vezes: pare com ValueError"


def testar_economia_nao_muda_o_que_recebeu():
    nota, y, ids = np.array(NOTA), pd.Series(Y), list(IDS)
    economia_da_lista(nota, y, ids, 3, 20.0, 100.0)
    assert list(nota) == NOTA and list(y) == Y and ids == IDS, "economia_da_lista não pode mudar o que recebeu"


def testar_numeros_da_parte_4():
    base = ler()
    _, ini, fim = TRIMESTRES[2]
    tr = treino_da_dobra(base, ini)
    me = base[(base["data_reserva"] >= ini) & (base["data_reserva"] < fim)]
    nota = nota_do_candidato("logística, antecedência em dias", tr, me)
    pegas, e = economia_da_lista(nota, me["cancelou"], me["id_reserva"], 140, 20.0, 100.0)
    assert (pegas, e) == (77, 4900.0), (
        f"na parte 4 da aula, a logística em out-dez pega 77 e economiza R$ 4.900 com 140 ligações; veio {(pegas, e)}")


def testar_escrita_no_proprio_arquivo():
    for fn, proibidos in ((treino_da_dobra, ("treino_ate",)), (economia_da_lista, ("lista(", "economia(", "medir("))):
        codigo = inspect.getsource(fn)
        corpo = codigo.split('"""', 2)[-1]
        usados = [p for p in proibidos if p in corpo]
        assert not usados, f"{fn.__name__} chama {', '.join(usados)} do case_viagens.py; escreva a conta no próprio arquivo"


def main():
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("testar_") and callable(f)]
    falhas = 0
    for nome, f in testes:
        try:
            f()
        except NotImplementedError:
            raise
        except AssertionError as e:
            falhas += 1
            print(f"FALHOU {nome}: {e}")
        except Exception as e:                      # erro que não é o ValueError pedido
            falhas += 1
            print(f"FALHOU {nome}: erro inesperado {type(e).__name__}: {e}")
    if falhas:
        raise SystemExit(f"{falhas} de {len(testes)} testes falharam")
    print(f"ok: {len(testes)} testes passaram")


if __name__ == "__main__":
    assert minha_comparacao.PRECO_NOVO == 35.0
    main()
