"""Ajusta uma regressão linear com várias colunas nos pedidos da Loja Exemplo.

Usa o pandas e o NumPy. O ajuste é a conta direta de mínimos quadrados, a mesma que a aula 02
chama de "conta direta": np.linalg.lstsq acha os coeficientes de menor MSE.

    from regressao import carregar, ajustar, residuos_por, intervalo

Nomes que você pode passar em `colunas`:

- qualquer coluna numérica do CSV, por exemplo `itens`, `cupom`, `cliente_novo`, `visitas_antes`;
- `categoria`: vira quatro colunas 0/1 (Casa, Eletro, Esporte, Papelaria). Beleza é a base;
- `canal`: vira a coluna `canal_app`, 1 para pedido feito no app. O site é a base;
- `itens_por_categoria`: o número de itens multiplicado por cada coluna de categoria, para a
  inclinação dos itens mudar de uma categoria para outra. Use junto com `itens` e `categoria`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
BASE_PADRAO = AQUI.parent / "dados" / "valor" / "pedidos_valor.csv"
COLUNAS_CSV = ["id_pedido", "data_pedido", "categoria", "itens", "peso_kg", "cliente_novo", "cupom",
               "canal", "visitas_antes", "prazo_prometido_dias", "hora_pedido", "dia_semana", "valor"]
CATEGORIAS = ["Beleza", "Casa", "Eletro", "Esporte", "Papelaria"]  # Beleza é a base
CANAIS = ["app", "site"]


def carregar(caminho=BASE_PADRAO):
    """Lê o CSV dos pedidos e para se faltar coluna, valor ou se aparecer categoria/canal estranho."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Rode de dentro da pasta da aula, com dados/ ao lado.")
    dados = pd.read_csv(caminho)
    faltam = [c for c in COLUNAS_CSV if c not in dados.columns]
    if faltam:
        raise ValueError(f"faltam colunas no CSV: {', '.join(faltam)}")
    vazios = dados[COLUNAS_CSV].isna().sum()
    vazios = vazios[vazios > 0]
    if len(vazios):
        raise ValueError("há valores vazios: " + ", ".join(f"{c} ({n})" for c, n in vazios.items()))
    estranhas = sorted(set(dados["categoria"]) - set(CATEGORIAS))
    if estranhas:
        raise ValueError(f"categoria que não está na lista: {', '.join(map(str, estranhas))}")
    canais = sorted(set(dados["canal"]) - set(CANAIS))
    if canais:
        raise ValueError(f"canal que não está na lista: {', '.join(map(str, canais))}")
    return dados


def matriz(dados, colunas):
    """Monta a tabela de números que entra no ajuste, com o intercepto na primeira coluna."""
    if not colunas:
        raise ValueError("passe pelo menos uma coluna")
    if len(set(colunas)) != len(colunas):
        raise ValueError("a mesma coluna aparece duas vezes na lista")
    partes = [pd.Series(1.0, index=dados.index, name="intercepto")]
    for nome in colunas:
        if nome == "categoria":
            for cat in CATEGORIAS[1:]:
                partes.append((dados["categoria"] == cat).astype(float).rename(f"categoria_{cat}"))
        elif nome == "canal":
            partes.append((dados["canal"] == "app").astype(float).rename("canal_app"))
        elif nome == "itens_por_categoria":
            if "itens" not in colunas or "categoria" not in colunas:
                raise ValueError("itens_por_categoria precisa de itens e categoria na mesma lista")
            for cat in CATEGORIAS[1:]:
                partes.append((dados["itens"] * (dados["categoria"] == cat)).astype(float)
                              .rename(f"itens_{cat}"))
        elif nome == "valor":
            raise ValueError("valor é o que o modelo prevê; ele não pode entrar como coluna")
        elif nome not in dados.columns:
            raise ValueError(f"coluna que não existe: {nome}")
        elif not pd.api.types.is_numeric_dtype(dados[nome]):
            raise ValueError(f"{nome} não é número; para texto, use categoria ou canal")
        else:
            partes.append(dados[nome].astype(float).rename(nome))
    return pd.concat(partes, axis=1)


class Ajuste:
    def __init__(self, coeficientes, previsto, residuo, n):
        self.coeficientes = coeficientes
        self.previsto = previsto
        self.residuo = residuo
        self.n = n


def ajustar(dados, colunas):
    """Ajusta valor = intercepto + soma(coeficiente × coluna) pelos mínimos quadrados."""
    X = matriz(dados, colunas)
    if len(X) <= X.shape[1]:
        raise ValueError(f"{len(X)} pedidos para {X.shape[1]} coeficientes: precisa de mais pedidos que coeficientes")
    A = X.to_numpy(float)
    for j in range(1, A.shape[1] + 1):
        if np.linalg.matrix_rank(A[:, :j]) < j:
            raise ValueError(f"a coluna {X.columns[j - 1]} repete o que as anteriores já dizem; "
                             "com ela, o ajuste não tem um só resultado")
    y = dados["valor"].to_numpy(float)
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    previsto = A.dot(b)
    return Ajuste(pd.Series(b, index=X.columns), pd.Series(previsto, index=dados.index),
                  pd.Series(y - previsto, index=dados.index), len(dados))


def faixa_de_itens(itens):
    return pd.cut(itens, [0, 1, 2, 3, 5, 12, 1000], labels=["1", "2", "3", "4 e 5", "6 a 12", "40 ou mais"])


def residuos_por(ajuste, dados, grupo):
    """Pedidos e resíduo médio por grupo. grupo pode ser o nome de uma coluna ou 'faixa_itens'."""
    chave = faixa_de_itens(dados["itens"]) if grupo == "faixa_itens" else dados[grupo]
    tabela = ajuste.residuo.groupby(chave, observed=True).agg(["count", "mean"])
    tabela.columns = ["pedidos", "resíduo médio"]
    return tabela


def intervalo(dados, colunas, vezes=500, sorteio=2026):
    """Reajusta o modelo em `vezes` sorteios dos pedidos, com reposição, e mostra onde caem
    os 95% do meio de cada coeficiente."""
    rng = np.random.default_rng(sorteio)
    X = matriz(dados, colunas)
    A = X.to_numpy(float)
    y = dados["valor"].to_numpy(float)
    todos = []
    for _ in range(vezes):
        i = rng.integers(0, len(A), len(A))
        todos.append(np.linalg.lstsq(A[i], y[i], rcond=None)[0])
    todos = np.array(todos)
    return pd.DataFrame({"de": np.percentile(todos, 2.5, axis=0),
                         "até": np.percentile(todos, 97.5, axis=0)}, index=X.columns)


def br(x, casas=2):
    """Número com vírgula decimal e ponto de milhar, como na aula."""
    if round(x, casas) == 0:
        x = 0.0
    s = f"{x:,.{casas}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")
