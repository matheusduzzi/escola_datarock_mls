"""Testes das funções tabela_calibracao e custo do meu_corte.py.

Rode de dentro da pasta da aula:  python3 testar_meu_corte.py
Os testes usam seis pedidos inventados aqui, não os pedidos de abril e maio: eles mostram que o
código faz a conta certa, e a leitura do Caio continua com você.
"""
from meu_corte import custo, tabela_calibracao

NOTA = [0.05, 0.10, 0.10, 0.30, 0.50, 0.90]
CANCELOU = [0, 0, 1, 0, 1, 1]


def perto(a, b):
    return abs(a - b) < 1e-9


def testar_faixas_e_medias():
    linhas = tabela_calibracao(NOTA, CANCELOU, [0, 0.1, 0.5, 1])
    assert len(linhas) == 3, f"eram 3 faixas com pedido, vieram {len(linhas)}"
    ini, fim, n, media, real = linhas[1]
    assert (perto(ini, 0.1), perto(fim, 0.5), n) == (True, True, 3), f"a faixa de 0,1 a 0,5 veio {linhas[1]}"
    assert perto(media, (0.10 + 0.10 + 0.30) / 3), f"nota média de 0,1 a 0,5 deu {media}"
    assert perto(real, 1 / 3), f"parte que cancelou de 0,1 a 0,5 deu {real}, era 1 de 3"


def testar_nota_no_limite_vai_para_cima():
    linhas = tabela_calibracao(NOTA, CANCELOU, [0, 0.1, 0.5, 1])
    contagens = [linha[2] for linha in linhas]
    assert contagens == [1, 3, 2], f"pedidos por faixa {contagens}; 0,10 e 0,50 entram na faixa de cima"


def testar_nota_um_entra_na_ultima():
    linhas = tabela_calibracao([0.2, 1.0], [0, 1], [0, 0.5, 1])
    assert [linha[2] for linha in linhas] == [1, 1], f"a nota 1 ficou fora da última faixa: {linhas}"


def testar_faixa_vazia_nao_entra():
    linhas = tabela_calibracao(NOTA, CANCELOU, [0, 0.02, 0.1, 1])
    assert [linha[0] for linha in linhas] == [0.02, 0.1], f"a faixa de 0 a 0,02 não tem pedido: {linhas}"


def testar_custo_com_mensagem():
    contatos, certos, total = custo(NOTA, CANCELOU, 0.1, 1, 15, evita=0.5)
    assert (contatos, certos) == (5, 3), f"com corte 0,1 deu {contatos} contatos e {certos} certos, eram 5 e 3"
    assert perto(total, 27.5), f"custo {total}; eram 5 mensagens + 3 × 7,50 da metade que não se evita"


def testar_custo_com_cancelamento_que_passa():
    contatos, certos, total = custo(NOTA, CANCELOU, 0.2, 1, 15, evita=0.5)
    assert (contatos, certos) == (3, 2), f"com corte 0,2 deu {contatos} contatos e {certos} certos, eram 3 e 2"
    assert perto(total, 33.0), f"custo {total}; eram 3 mensagens + 2 × 7,50 + 1 cancelamento de 15"


def testar_custo_evita_tudo():
    _, _, total = custo(NOTA, CANCELOU, 0.1, 5, 15)
    assert perto(total, 25.0), f"custo {total}; com evita = 1 são só as 5 ligações de R$ 5"


def testar_tamanhos_diferentes():
    for funcao, args in ((tabela_calibracao, (NOTA, [0, 1], [0, 1])), (custo, (NOTA, [0, 1], 0.1, 1, 15))):
        try:
            funcao(*args)
        except ValueError:
            continue
        raise AssertionError(f"{funcao.__name__} aceitou 6 notas e 2 resultados; tinha que parar com ValueError")


def testar_nota_fora_de_zero_a_um():
    try:
        tabela_calibracao([0.2, 1.3], [0, 1], [0, 0.5, 1])
    except ValueError:
        return
    raise AssertionError("tabela_calibracao aceitou a nota 1,3; tinha que parar com ValueError")


def testar_limites_invalidos():
    for limites in ([0.1, 0.5, 1], [0, 0.5, 0.9], [0, 0.5, 0.3, 1]):
        try:
            tabela_calibracao(NOTA, CANCELOU, limites)
        except ValueError:
            continue
        raise AssertionError(f"tabela_calibracao aceitou os limites {limites}; tinha que parar com ValueError")


def testar_evita_invalido():
    for evita in (0, -0.5, 1.5):
        try:
            custo(NOTA, CANCELOU, 0.1, 1, 15, evita=evita)
        except ValueError:
            continue
        raise AssertionError(f"custo aceitou evita = {evita}; tinha que parar com ValueError")


if __name__ == "__main__":
    import sys

    testes = [(nome, f) for nome, f in sorted(globals().items()) if nome.startswith("testar_")]
    falhas = 0
    for nome, teste in testes:
        try:
            teste()
        except AssertionError as erro:
            falhas += 1
            print(f"FALHOU {nome}: {erro}")
        except Exception as erro:
            falhas += 1
            print(f"FALHOU {nome}: {type(erro).__name__}: {erro}")
    if falhas:
        print(f"{falhas} de {len(testes)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(testes)} testes passaram")
