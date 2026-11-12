"""Escolher o alfa do Ridge nas dobras dos pedidos de janeiro a maio e abrir junho uma vez só.

Rode de dentro da pasta da aula:

    python3 experimento.py               # escolhe o alfa nas dobras e grava experimento.json
    python3 experimento.py --sorteios    # a mesma escolha em 20 sorteios das dobras; não grava nada
    python3 experimento.py --abrir-teste # refaz a escolha, compara com o registro e mede junho, uma vez

Rodar `python3 experimento.py` de novo, com o registro já gravado, compara número por número.
Igual, ele diz igual. Diferente, ele lista o que mudou e sai com código 1.

O custo de cada ajuste é o mesmo da aula «Ridge, Lasso e Elastic Net», com as colunas padronizadas:

    MSE + l1 × soma(|coeficiente|) + l2 × soma(coeficiente²)

No Ridge, o alfa é o l2. No Lasso, o alfa é o l1. O intercepto fica fora da cobrança.
Junho só é lido dentro de abrir_teste(): nenhuma conta da escolha vê esses pedidos.
"""
import hashlib
import json
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
ARQUIVO = AQUI.parent / "dados" / "valor" / "pedidos_valor.csv"
REGISTRO = AQUI / "experimento.json"
FIM_DO_TREINO = "2026-06-01"          # treino: pedidos antes dessa data; teste: junho
K = 5
SORTEIO = 14
GRADE_RIDGE = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
REGRA = "menor MSE médio nas 5 dobras; empate exato fica com o maior alfa"
CATEGORIAS = ["Eletro", "Casa", "Papelaria", "Esporte", "Beleza"]
DO_PEDIDO = ["cliente_novo", "cupom", "visitas_antes", "prazo_prometido_dias", "hora_pedido", "dia_semana"]
COLUNAS_AULA_10 = ["itens", "peso_kg"] + DO_PEDIDO
COLUNAS = (DO_PEDIDO + [f"cat_{c}" for c in CATEGORIAS[1:]] + ["app"]
           + [f"itens_{c}" for c in CATEGORIAS] + [f"peso_{c}" for c in CATEGORIAS])


# ---------- dados ----------

def _ler(caminho=ARQUIVO):
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com ../dados/ ao lado.")
    dados = pd.read_csv(caminho)
    faltam = [c for c in ["id_pedido", "data_pedido", "categoria", "canal", "itens", "peso_kg", "valor"] + DO_PEDIDO
              if c not in dados.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    varejo = dados[dados["itens"] < 40]          # os 12 pedidos de empresa ficam de fora, como na aula 03
    return varejo.sort_values(["data_pedido", "id_pedido"]).reset_index(drop=True)


def digital(caminho=ARQUIVO):
    """O sha256 do arquivo: muda se uma letra do CSV mudar."""
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def carregar_treino(caminho=ARQUIVO):
    """Os pedidos de varejo de janeiro a maio. Junho fica de fora aqui."""
    dados = _ler(caminho)
    return dados[dados["data_pedido"] < FIM_DO_TREINO].reset_index(drop=True)


def _carregar_teste(caminho=ARQUIVO):
    dados = _ler(caminho)
    return dados[dados["data_pedido"].str.startswith("2026-06-")].reset_index(drop=True)


def montar_colunas(dados):
    """As 21 colunas: 6 do pedido, 4 de categoria (Eletro é a referência), o canal app e itens e peso por categoria.

    A lista de categorias é fixa, escrita aqui. Ela não sai dos dados, então não depende de quem está no treino.
    """
    x = dados[DO_PEDIDO].astype(float).copy()
    for c in CATEGORIAS[1:]:
        x[f"cat_{c}"] = (dados["categoria"] == c).astype(float)
    x["app"] = (dados["canal"] == "app").astype(float)
    for c in CATEGORIAS:
        x[f"itens_{c}"] = dados["itens"] * (dados["categoria"] == c)
    for c in CATEGORIAS:
        x[f"peso_{c}"] = dados["peso_kg"] * (dados["categoria"] == c)
    return x[COLUNAS].astype(float)


# ---------- as contas da aula 10 ----------

def padronizar(ajuste, outros):
    """Tira a média e divide pelo desvio, os dois calculados só nos pedidos de ajuste."""
    media = ajuste.mean()
    desvio = ajuste.std(ddof=0)
    if (desvio == 0).any():
        raise ValueError(f"coluna sem variação no ajuste: {', '.join(desvio.index[desvio == 0])}")
    return (ajuste - media) / desvio, (outros - media) / desvio, media, desvio


def ajustar(z, y, l1=0.0, l2=0.0):
    """Coeficientes nas colunas padronizadas e o intercepto (o mesmo ajuste da aula 10)."""
    if l1 < 0 or l2 < 0:
        raise ValueError("as penalidades não podem ser negativas")
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = z.shape
    intercepto = y.mean()
    yc = y - intercepto
    if l1 == 0:
        a = z.T.dot(z) / n + l2 * np.eye(p)
        if np.linalg.matrix_rank(a) < p:
            raise ValueError("colunas repetidas e sem penalidade: o ajuste não tem um só resultado")
        return np.linalg.solve(a, z.T.dot(yc) / n), intercepto
    var = (z ** 2).mean(axis=0)
    b = np.zeros(p)
    r = yc.copy()
    for _ in range(20000):
        maior = 0.0
        for j in range(p):
            r += z[:, j] * b[j]
            c = np.mean(z[:, j] * r)
            novo = np.sign(c) * max(abs(c) - l1 / 2, 0.0) / (var[j] + l2)
            r -= z[:, j] * novo
            maior = max(maior, abs(novo - b[j]))
            b[j] = novo
        if maior < 1e-10:
            return b, intercepto
    raise RuntimeError("a descida por coordenada não parou em 20000 voltas")


def mse(z, y, coef, intercepto):
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((y - intercepto - z.dot(coef)) ** 2))


# ---------- as dobras ----------

def dividir_em_dobras(n, k, sorteio):
    """A dobra (de 0 a k − 1) de cada um dos n pedidos.

    Embaralha as posições 0 a n − 1 com np.random.default_rng(sorteio).permutation(n). A posição que sai
    em i-ésimo lugar fica na dobra i % k. Assim as dobras têm o mesmo tamanho, com diferença de no máximo 1.
    """
    if k < 2 or k > n:
        raise ValueError(f"o número de dobras vai de 2 até o número de pedidos ({n}); veio {k}")
    ordem = np.random.default_rng(sorteio).permutation(n)
    dobras = np.empty(n, dtype=int)
    dobras[ordem] = np.arange(n) % k
    return dobras


def erro_nas_dobras(x, y, dobras, l1=0.0, l2=0.0):
    """O MSE em cada dobra: ajusta nas outras dobras, com a padronização feita nelas, e mede nessa."""
    y = np.asarray(y, dtype=float)
    dobras = np.asarray(dobras)
    if len(dobras) != len(x) or len(y) != len(x):
        raise ValueError(f"tamanhos diferentes: {len(x)} pedidos, {len(y)} valores, {len(dobras)} dobras")
    k = int(dobras.max()) + 1
    if sorted(set(dobras.tolist())) != list(range(k)):
        raise ValueError("as dobras precisam ser 0, 1, ..., k − 1, todas com pedido")
    erros = []
    for j in range(k):
        fora = dobras != j
        z, zv, _, _ = padronizar(x[fora], x[~fora])
        coef, intercepto = ajustar(z, y[fora], l1, l2)
        erros.append(mse(zv, y[~fora], coef, intercepto))
    return erros


# ---------- a escolha ----------

def escolher(medias, grade):
    """A regra: menor MSE médio; empate exato fica com o maior alfa."""
    menor = min(medias)
    return max(a for a, m in zip(grade, medias) if m == menor)


def penalidades(metodo, alfa):
    if metodo == "Ridge":
        return 0.0, float(alfa)
    if metodo == "Lasso":
        return float(alfa), 0.0
    raise ValueError(f"método desconhecido: {metodo}")


def experimento(metodo, grade, sorteio=SORTEIO, dividir=dividir_em_dobras, erros=erro_nas_dobras,
                caminho=ARQUIVO):
    """Roda a escolha inteira só com janeiro a maio e devolve o registro, sem gravar."""
    treino = carregar_treino(caminho)
    x, y = montar_colunas(treino), treino["valor"].to_numpy(float)
    dobras = np.asarray(dividir(len(treino), K, sorteio))
    resultados = {}
    for alfa in grade:
        resultados[str(alfa)] = [float(e) for e in erros(x, y, dobras, *penalidades(metodo, alfa))]
    medias = [float(np.mean(resultados[str(a)])) for a in grade]
    alfa = escolher(medias, grade)
    referencia = [float(e) for e in erros(treino[COLUNAS_AULA_10].astype(float), y, dobras, 0.0, 0.0)]
    zeros = {}
    for a in grade:
        z, _, _, _ = padronizar(x, x)
        coef, _ = ajustar(z, y, *penalidades(metodo, a))
        zeros[str(a)] = [c for c, b in zip(COLUNAS, coef) if b == 0]
    ids = treino["id_pedido"].to_numpy()
    texto_dobras = ";".join(",".join(str(i) for i in sorted(ids[dobras == j])) for j in range(K))
    return {
        "pergunta": f"alfa do {metodo} para o valor do pedido",
        "dados": {"arquivo": "dados/valor/pedidos_valor.csv", "sha256": digital(caminho),
                  "filtro": "itens < 40 (varejo)", "treino": "2026-01-01 a 2026-05-31",
                  "pedidos_treino": int(len(treino)),
                  "valor_medio_por_mes": {m: float(v) for m, v in
                                          treino.groupby(treino["data_pedido"].str[5:7])["valor"].mean().items()},
                  "teste": "junho de 2026, fechado até abrir_teste"},
        "colunas": COLUNAS,
        "dobras": {"k": K, "sorteio": sorteio, "tamanhos": [int((dobras == j).sum()) for j in range(K)],
                   "maior_valor": [float(y[dobras == j].max()) for j in range(K)],
                   "sha256_dos_ids": hashlib.sha256(texto_dobras.encode()).hexdigest()},
        "metodo": metodo,
        "grade": grade,
        "regra": REGRA,
        "resultados": resultados,
        "referencia_8_colunas_sem_penalidade": referencia,
        "colunas_em_zero_no_treino": zeros,
        "escolha": {"alfa": alfa, "mse_medio": float(np.mean(resultados[str(alfa)]))},
        "versoes": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "registrado_em": date.today().isoformat(),
        "teste": None,
    }


# ---------- comparar com o registro ----------

FORA_DA_COMPARACAO = ("versoes", "registrado_em", "teste")


def diferencas(registro, agora, caminho=""):
    """Lista o que mudou entre o registro e a rodada de agora; números iguais até 6 casas valem como iguais."""
    if isinstance(registro, dict) and isinstance(agora, dict):
        saida = []
        for chave in sorted(set(registro) | set(agora)):
            if not caminho and chave in FORA_DA_COMPARACAO:
                continue
            nome = f"{caminho}.{chave}" if caminho else chave
            if chave not in registro or chave not in agora:
                saida.append(f"{nome}: só existe {'agora' if chave in agora else 'no registro'}")
            else:
                saida += diferencas(registro[chave], agora[chave], nome)
        return saida
    if isinstance(registro, list) and isinstance(agora, list) and len(registro) == len(agora):
        return [d for i, (a, b) in enumerate(zip(registro, agora)) for d in diferencas(a, b, f"{caminho}[{i}]")]
    if isinstance(registro, (int, float)) and isinstance(agora, (int, float)) \
            and not isinstance(registro, bool) and not isinstance(agora, bool):
        return [] if round(float(registro), 6) == round(float(agora), 6) else [f"{caminho}: registro {registro}, agora {agora}"]
    return [] if registro == agora else [f"{caminho}: registro {registro}, agora {agora}"]


def avisar_versoes(registro):
    agora = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for nome, versao in registro.get("versoes", {}).items():
        if agora.get(nome) != versao:
            print(f"aviso: o registro foi feito com {nome} {versao}; aqui é {agora.get(nome)}")


# ---------- números com vírgula ----------

def br(x, casas=0):
    if round(x, casas) == 0:
        x = 0.0
    s = f"{x:,.{casas}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def alfa_br(a):
    return br(a, max(0, len(repr(float(a)).split(".")[1].rstrip("0")))) if a else "0"


# ---------- o que o programa mostra ----------

def mostrar(reg):
    d, dob = reg["dados"], reg["dobras"]
    print(f"treino: {d['pedidos_treino']} pedidos de varejo de janeiro a maio de 2026; junho fica fechado")
    print(f"dados: {d['arquivo']}, sha256 {d['sha256'][:12]}")
    nomes = {"01": "jan", "02": "fev", "03": "mar", "04": "abr", "05": "mai"}
    print("valor médio por mês: " + ", ".join(f"{nomes.get(m, m)} R$ {br(v, 2)}" for m, v in d["valor_medio_por_mes"].items()))
    print(f"colunas: {len(reg['colunas'])} (6 do pedido, 4 de categoria, o canal app, itens e peso por categoria)")
    print()
    print(f"1. As dobras: {dob['k']}, pelo sorteio {dob['sorteio']}")
    print("  dobra   " + "".join(f"{j + 1:>8}" for j in range(dob["k"])))
    print("  pedidos " + "".join(f"{t:>8}" for t in dob["tamanhos"]))
    print("  maior R$" + "".join(f"{br(v):>8}" for v in dob["maior_valor"]))
    print(f"  cada pedido fica em uma dobra só; o código das dobras é {dob['sha256_dos_ids'][:12]}")
    print()
    print(f"2. MSE de cada alfa em cada dobra ({reg['metodo']}, padronizado dentro da dobra)")
    print(f"  {'alfa':8}" + "".join(f"{'dobra ' + str(j + 1):>9}" for j in range(dob["k"]))
          + f"{'média':>9}{'desvio':>8}{'zeros':>7}")
    for a in reg["grade"]:
        e = reg["resultados"][str(a)]
        print(f"  {alfa_br(a):8}" + "".join(f"{br(v):>9}" for v in e) + f"{br(np.mean(e)):>9}"
              + f"{br(np.std(e)):>8}{len(reg['colunas_em_zero_no_treino'][str(a)]):>7}")
    ref = reg["referencia_8_colunas_sem_penalidade"]
    print(f"  {'8 colunas da aula 10, sem penalidade':44}{br(np.mean(ref)):>9}{br(np.std(ref)):>8}")
    print("  zeros: colunas com coeficiente zero no ajuste com os pedidos de janeiro a maio")
    print()
    alfa = reg["escolha"]["alfa"]
    escolhido = np.array(reg["resultados"][str(alfa)])
    print("3. A escolha")
    print(f"  regra, escrita antes de rodar: {reg['regra']}")
    print(f"  escolhido: alfa {alfa_br(alfa)}, MSE médio {br(reg['escolha']['mse_medio'])}")
    if alfa != 0 and "0" in reg["resultados"]:
        zero = np.array(reg["resultados"]["0"])
        print(f"  contra o alfa 0, dobra a dobra: menor em {int((escolhido < zero).sum())} de {len(zero)}; "
              f"diferenças {', '.join(br(v) for v in escolhido - zero)}; média {br(np.mean(escolhido - zero))}")
    print(f"  de uma dobra para outra, o mesmo alfa vai de {br(escolhido.min())} a {br(escolhido.max())}")
    print()


def rodar_escolha(registro, metodo, grade, dividir, erros):
    reg = experimento(metodo, grade, SORTEIO, dividir, erros)
    mostrar(reg)
    print("4. O registro")
    if not registro.exists():
        registro.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        n = sum(len(v) for v in reg["resultados"].values())
        print(f"  gravei {registro.name}: dados, colunas, dobras, grade, regra, os {n} MSE, a escolha e as versões")
        print("  junho continua fechado")
        return 0
    antigo = json.loads(registro.read_text(encoding="utf-8"))
    avisar_versoes(antigo)
    mudou = diferencas(antigo, reg)
    if mudou:
        print(f"  diferente de {registro.name} em {len(mudou)} {'ponto' if len(mudou) == 1 else 'pontos'}:")
        for linha in mudou[:20]:
            print(f"    {linha}")
        return 1
    n = sum(len(v) for v in reg["resultados"].values())
    print(f"  igual a {registro.name}: os mesmos dados, as mesmas dobras e os {n} MSE")
    if antigo.get("teste"):
        print(f"  junho já foi aberto para esse registro: MSE {br(antigo['teste']['mse'])}")
    else:
        print("  junho continua fechado")
    return 0


def abrir_teste(registro, metodo, grade, dividir, erros):
    """Refaz a escolha, exige que ela bata com o registro e só então mede junho, uma vez."""
    if not registro.exists():
        print(f"não achei {registro.name}. Rode a escolha antes (sem --abrir-teste): o teste só abre depois "
              "da escolha gravada")
        return 1
    antigo = json.loads(registro.read_text(encoding="utf-8"))
    avisar_versoes(antigo)
    reg = experimento(metodo, grade, antigo["dobras"]["sorteio"], dividir, erros)
    mudou = diferencas(antigo, reg)
    if mudou:
        print(f"refiz a escolha, e ela não bate com {registro.name} em {len(mudou)} {'ponto' if len(mudou) == 1 else 'pontos'}:")
        for linha in mudou[:20]:
            print(f"  {linha}")
        print("o teste não abre com um registro que não bate com a conta")
        return 1
    if antigo.get("teste"):
        t = antigo["teste"]
        print(f"junho já foi aberto para esse registro em {t['aberto_em']}: MSE {br(t['mse'])} "
              f"com o alfa {alfa_br(t['alfa'])}")
        print("o teste abre uma vez. Para outra pergunta, comece outro registro")
        return 1
    print(f"refiz a escolha e comparei com {registro.name}: igual")
    alfa = antigo["escolha"]["alfa"]
    treino = carregar_treino()
    teste = _carregar_teste()
    x, y = montar_colunas(treino), treino["valor"].to_numpy(float)
    z, zt, media, desvio = padronizar(x, montar_colunas(teste))
    coef, intercepto = ajustar(z, y, *penalidades(metodo, alfa))
    pago = teste["valor"].to_numpy(float)
    previsto = intercepto + np.asarray(zt, dtype=float).dot(coef)
    erro = mse(zt, pago, coef, intercepto)
    zeros = [c for c, b in zip(COLUNAS, coef) if b == 0]
    print(f"ajuste final: {metodo} alfa {alfa_br(alfa)} nos {len(treino)} pedidos de janeiro a maio")
    print(f"junho, aberto agora: {len(teste)} pedidos, valor médio R$ {br(teste['valor'].mean(), 2)}")
    print(f"  MSE {br(erro)} | raiz do MSE R$ {br(np.sqrt(erro), 2)}")
    print(f"  nas dobras, o mesmo alfa deu MSE médio {br(antigo['escolha']['mse_medio'])} "
          f"(raiz R$ {br(np.sqrt(antigo['escolha']['mse_medio']), 2)})")
    print(f"  soma do mês: previsto R$ {br(previsto.sum(), 2)}, pago R$ {br(pago.sum(), 2)}")
    print(f"  colunas com coeficiente zero: {', '.join(zeros) if zeros else 'nenhuma'}")
    antigo["teste"] = {"aberto_em": date.today().isoformat(), "alfa": alfa, "pedidos": int(len(teste)),
                       "mse": erro, "raiz_do_mse": float(np.sqrt(erro)), "colunas_em_zero": zeros,
                       "soma_prevista": float(previsto.sum()), "soma_paga": float(pago.sum()),
                       "coeficientes_em_reais_por_unidade": {c: float(b / desvio[c]) for c, b in zip(COLUNAS, coef)},
                       "intercepto_em_reais": float(intercepto - sum(b * media[c] / desvio[c]
                                                                     for c, b in zip(COLUNAS, coef)))}
    registro.write_text(json.dumps(antigo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"gravei o teste em {registro.name}; junho não abre de novo para esse registro")
    return 0


def sorteios(metodo, grade, dividir, erros, quantos=20):
    print(f"A mesma escolha em {quantos} sorteios das dobras (0 a {quantos - 1}); ensaio, não grava nada")
    print(f"  {'sorteio':>7}  {'alfa escolhido':>14}  {'MSE médio':>9}")
    escolhas = []
    for s in range(quantos):
        reg = experimento(metodo, grade, s, dividir, erros)
        escolhas.append((reg["escolha"]["alfa"], reg["escolha"]["mse_medio"]))
        print(f"  {s:>7}  {alfa_br(escolhas[-1][0]):>14}  {br(escolhas[-1][1]):>9}")
    contagem = ", ".join(f"{alfa_br(a)} em {sum(1 for e, _ in escolhas if e == a)}"
                         for a in grade if any(e == a for e, _ in escolhas))
    print(f"  alfa escolhido: {contagem}")
    print(f"  MSE médio da escolha: de {br(min(m for _, m in escolhas))} a {br(max(m for _, m in escolhas))}")
    return 0


def rodar(argv, registro=REGISTRO, metodo="Ridge", grade=GRADE_RIDGE, dividir=dividir_em_dobras,
          erros=erro_nas_dobras):
    if argv not in ([], ["--sorteios"], ["--abrir-teste"]):
        print(f"use: python3 {Path(sys.argv[0]).name} [--sorteios | --abrir-teste]")
        return 2
    if argv == ["--sorteios"]:
        return sorteios(metodo, grade, dividir, erros)
    if argv == ["--abrir-teste"]:
        return abrir_teste(Path(registro), metodo, grade, dividir, erros)
    return rodar_escolha(Path(registro), metodo, grade, dividir, erros)


if __name__ == "__main__":
    sys.exit(rodar(sys.argv[1:]))
