"""Treina o modelo de produção e gera o relatório de avaliação da monografia.

Saídas:
  modelo/modelo_fuzzy_artmap.json   modelo publicado pela API (comitê de redes)
  resultados/avaliacao.md           tabelas comparativas e explicação dos números
  resultados/*.png                  figuras para a defesa

Uso: python treinar_modelo.py [--sem-figuras]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from fuzzy_artmap import ComiteFuzzyARTMAP, FuzzyARTMAP
from gerar_dados_ficticios import teto_bayes
from servico_risco import carregar_base_csv, treinar_servico

PASTA = Path(__file__).resolve().parent
ARQUIVO_BASE = PASTA / "dados" / "dados_alunos_tcc.csv"
ARQUIVO_MODELO = PASTA / "modelo" / "modelo_fuzzy_artmap.json"
PASTA_RESULTADOS = PASTA / "resultados"

N_REDES = 7
SEMENTE = 42
N_ORDENS = 9  # ordem do CSV + 8 permutações (rede única)
SEMENTES_COMITE = [42, 43, 44, 45, 46]


def pct(valor: float, casas: int = 1) -> str:
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def metricas(y, previsto, score=None) -> dict:
    return {
        "acuracia": accuracy_score(y, previsto),
        "precisao": precision_score(y, previsto, zero_division=0),
        "recall": recall_score(y, previsto, zero_division=0),
        "f1": f1_score(y, previsto, zero_division=0),
        "auc": roc_auc_score(y, score) if score is not None else None,
    }


def resumir(nome: str, execucoes: list[dict], observacao: str = "") -> dict:
    """Média (e desvio da acurácia) de várias execuções do mesmo modelo."""
    linha = {"modelo": nome, "observacao": observacao, "execucoes": len(execucoes)}
    for chave in ("acuracia", "precisao", "recall", "f1", "auc"):
        valores = [e[chave] for e in execucoes if e[chave] is not None]
        linha[chave] = float(np.mean(valores)) if valores else None
    acuracias = [e["acuracia"] for e in execucoes]
    linha["desvio"] = float(np.std(acuracias)) if len(acuracias) > 1 else None
    linha["acuracias"] = acuracias
    return linha


def cv_rede_unica(X, y, folds, ordens: int, **parametros) -> list[dict]:
    """Uma rede por dobra; a 1ª execução usa a ordem do CSV, as demais permutações."""
    rng = np.random.default_rng(0)
    execucoes = []
    for o in range(ordens):
        previsto = np.zeros_like(y)
        for treino, teste in folds:
            ordem = treino if o == 0 else rng.permutation(treino)
            rede = FuzzyARTMAP(**parametros).treinar(X, y, ordem)
            previsto[teste] = rede.prever_lote(X[teste])
        execucoes.append(metricas(y, previsto))
    return execucoes


def cv_comite(X, y, folds, semente: int):
    previsto, score = np.zeros_like(y), np.zeros(len(y))
    acc_dobras = []
    for treino, teste in folds:
        comite = ComiteFuzzyARTMAP(n_redes=N_REDES, semente=semente).treinar(X[treino], y[treino])
        score[teste] = comite.score_lote(X[teste])
        previsto[teste] = (score[teste] >= 0.5).astype(int)
        acc_dobras.append(accuracy_score(y[teste], previsto[teste]))
    return metricas(y, previsto, score), previsto, acc_dobras


def cv_sklearn(modelo, X, y, folds) -> dict:
    previsto, score = np.zeros_like(y), np.zeros(len(y))
    for treino, teste in folds:
        modelo.fit(X[treino], y[treino])
        previsto[teste] = modelo.predict(X[teste])
        score[teste] = modelo.predict_proba(X[teste])[:, 1]
    tem_score = not isinstance(modelo, DummyClassifier)
    return metricas(y, previsto, score if tem_score else None)


# ---------------------------------------------------------------------------
# Figuras (estáticas, para impressão): ênfase no modelo do TCC, grade discreta
# ---------------------------------------------------------------------------
COR = {
    "superficie": "#fcfcfb",
    "tinta": "#0b0b0b",
    "tinta_secundaria": "#52514e",
    "tinta_suave": "#898781",
    "grade": "#e1e0d9",
    "base": "#c3c2b7",
    "destaque": "#2a78d6",
}
FUNDO_ROTULO = {"boxstyle": "square,pad=0.15", "facecolor": COR["superficie"], "edgecolor": "none"}
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _estilo(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "axes.facecolor": COR["superficie"],
            "figure.facecolor": COR["superficie"],
            "axes.edgecolor": COR["base"],
            "axes.labelcolor": COR["tinta_secundaria"],
            "xtick.color": COR["tinta_suave"],
            "ytick.color": COR["tinta_secundaria"],
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": COR["tinta"],
            "axes.titlelocation": "left",
        }
    )


def _eixo_limpo(ax, grade_em="x") -> None:
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COR["base"])
    ax.tick_params(length=0)
    ax.grid(axis=grade_em, color=COR["grade"], linewidth=0.8)
    ax.set_axisbelow(True)


def figura_comparacao(linhas: list[dict], teto: float, destino: Path, plt) -> None:
    nomes = [l["modelo"] for l in linhas]
    destaque = [l["modelo"].startswith("Comitê") for l in linhas]
    cores = [COR["destaque"] if d else COR["base"] for d in destaque]
    y = np.arange(len(linhas))[::-1]
    fig, eixos = plt.subplots(1, 2, figsize=(9.5, 0.42 * len(linhas) + 1.3), sharey=True)
    for ax, chave, titulo in ((eixos[0], "acuracia", "Acurácia"), (eixos[1], "recall", "Recall (alunos que evadiram)")):
        valores = [l[chave] for l in linhas]
        ax.barh(y, valores, height=0.32, color=cores)
        for yi, v, d in zip(y, valores, destaque):
            ax.text(v + 0.012, yi, pct(v), va="center", fontsize=8.5, zorder=3, bbox=FUNDO_ROTULO,
                    color=COR["tinta"] if d else COR["tinta_secundaria"], fontweight="semibold" if d else "normal")
        ax.set_xlim(0, 1.13)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(titulo, pad=10)
        _eixo_limpo(ax)
    eixos[0].axvline(teto, color=COR["tinta_secundaria"], linewidth=1.2, zorder=2)
    eixos[0].text(teto, len(linhas) - 0.35, f"teto teórico {pct(teto)}", ha="center", va="bottom",
                  fontsize=8, color=COR["tinta_secundaria"])
    eixos[0].set_yticks(y, nomes)
    fig.suptitle("Modelos comparados na mesma validação cruzada (5 dobras, base sintética)",
                 x=0.01, ha="left", fontsize=9, color=COR["tinta_secundaria"])
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def figura_sensibilidade(grupos: list[tuple[str, list[float]]], destino: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    for i, (nome, acuracias) in enumerate(reversed(grupos)):
        acuracias = np.array(acuracias)
        ax.plot([acuracias.min(), acuracias.max()], [i, i], color=COR["base"], linewidth=2, solid_capstyle="round", zorder=1)
        ax.scatter(acuracias, np.full(len(acuracias), i), s=42, color=COR["destaque"],
                   edgecolors=COR["superficie"], linewidths=1.2, zorder=2)
        ax.text(acuracias.max() + 0.004, i, f"{pct(acuracias.min())} a {pct(acuracias.max())}", va="center",
                fontsize=8.5, color=COR["tinta_secundaria"])
    ax.set_yticks(range(len(grupos)), [g[0] for g in reversed(grupos)])
    ax.set_ylim(-0.6, len(grupos) - 0.4)
    minimo = min(min(g[1]) for g in grupos)
    inicio = np.floor(minimo * 20) / 20
    ax.set_xlim(inicio - 0.01, 0.93)
    ax.set_xticks(np.arange(inicio, 0.901, 0.05), [pct(v, 0) for v in np.arange(inicio, 0.901, 0.05)])
    ax.set_title("Acurácia ao mudar só a ordem de treino (rede única) ou a semente (comitê)", pad=10)
    _eixo_limpo(ax)
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def figura_matriz(matriz: np.ndarray, destino: Path, plt) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    mapa = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
    ax.imshow(matriz, cmap=mapa, vmin=0, vmax=matriz.max())
    rotulos = ["Não evadiu", "Evadiu"]
    for i in range(2):
        for j in range(2):
            fracao = matriz[i, j] / matriz[i].sum()
            escuro = matriz[i, j] > 0.55 * matriz.max()
            ax.text(j, i, f"{matriz[i, j]}\n{pct(fracao)} da linha", ha="center", va="center", fontsize=9,
                    color="#ffffff" if escuro else COR["tinta"], fontweight="semibold")
    ax.set_xticks([0, 1], rotulos)
    ax.set_yticks([0, 1], rotulos)
    ax.set_xlabel("Previsão do comitê")
    ax.set_ylabel("Desfecho real")
    ax.set_title("Matriz de confusão (5 dobras)", pad=10)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def figura_dobras(acc_dobras: list[float], destino: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(acc_dobras))
    ax.bar(x, acc_dobras, width=0.32, color=COR["destaque"])
    for xi, v in zip(x, acc_dobras):
        ax.text(xi, v + 0.015, pct(v), ha="center", va="bottom", fontsize=8.5, color=COR["tinta"])
    media = float(np.mean(acc_dobras))
    ax.set_xticks(x, [f"Dobra {i + 1}" for i in x])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: pct(v, 0))
    ax.set_title(f"Acurácia por dobra — comitê de {N_REDES} redes (média {pct(media)})", pad=10)
    _eixo_limpo(ax, grade_em="y")
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def tabela_markdown(linhas: list[dict]) -> str:
    def fmt(v):
        return "—" if v is None else pct(v)

    saida = ["| Modelo | Acurácia | Precisão | Recall (evadiu) | F1 | AUC | Observação |", "|---|---|---|---|---|---|---|"]
    for l in linhas:
        desvio = f" ± {l['desvio'] * 100:.1f}".replace(".", ",") + " p.p." if l["desvio"] else ""
        acc = fmt(l["acuracia"]) + desvio
        f1 = f"{l['f1']:.3f}".replace(".", ",")
        auc = "—" if l["auc"] is None else f"{l['auc']:.3f}".replace(".", ",")
        nome = f"**{l['modelo']}**" if l["modelo"].startswith("Comitê") else l["modelo"]
        saida.append(f"| {nome} | {acc} | {fmt(l['precisao'])} | {fmt(l['recall'])} | {f1} | {auc} | {l['observacao']} |")
    return "\n".join(saida)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo e gera o relatório de avaliação.")
    parser.add_argument("--sem-figuras", action="store_true")
    args = parser.parse_args()

    X, y = carregar_base_csv(ARQUIVO_BASE)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE).split(X, y))
    teto = teto_bayes()
    print(f"Base: {len(y)} alunos, {y.mean():.1%} evadiram. Teto teórico: {teto:.1%}")

    print("Avaliando Fuzzy ARTMAP (versão anterior, cada correção isolada e versão corrigida)...")
    anterior = cv_rede_unica(X, y, folds, N_ORDENS, aprender_antes_do_map_field=True, predicao_versao_anterior=True)
    so_aprendizado = cv_rede_unica(X, y, folds, N_ORDENS, aprender_antes_do_map_field=True)
    so_predicao = cv_rede_unica(X, y, folds, N_ORDENS, predicao_versao_anterior=True)
    corrigido = cv_rede_unica(X, y, folds, N_ORDENS)
    decomposicao = [
        resumir("Versão anterior (os dois defeitos)", anterior),
        resumir("Só o defeito 2 corrigido (predição)", so_aprendizado),
        resumir("Só o defeito 1 corrigido (aprendizado)", so_predicao),
        resumir("Os dois corrigidos", corrigido),
    ]

    print(f"Avaliando o comitê de {N_REDES} redes...")
    execucoes_comite, matriz, acc_dobras = [], None, None
    for semente in SEMENTES_COMITE:
        m, previsto, dobras = cv_comite(X, y, folds, semente)
        execucoes_comite.append(m)
        if semente == SEMENTE:
            metricas_producao, matriz, acc_dobras = m, confusion_matrix(y, previsto), dobras

    print("Avaliando modelos de comparação...")
    baselines = [
        ("Chutar 'ninguém evade'", DummyClassifier(strategy="most_frequent"), "referência mínima"),
        ("Regressão logística", LogisticRegression(), ""),
        ("Árvore de decisão (prof. 3)", DecisionTreeClassifier(max_depth=3, random_state=0), ""),
        ("Random Forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0), ""),
    ]
    regra = ((X[:, 0] < 0.70) | (X[:, 1] < 0.45)).astype(int)

    linhas = [resumir(nome, [cv_sklearn(modelo, X, y, folds)], obs) for nome, modelo, obs in baselines]
    linhas.append(resumir("Regra fixa: freq < 70% ou nota < 4,5", [metricas(y, regra)], "sem aprendizado"))
    linhas.append(resumir("Fuzzy ARTMAP — versão anterior", anterior, f"implementação original; {N_ORDENS} ordens"))
    linhas.append(resumir("Fuzzy ARTMAP — corrigido (1 rede)", corrigido, f"{N_ORDENS} ordens de treino"))
    linhas.append(resumir(f"Comitê de {N_REDES} redes (produção)", execucoes_comite,
                          f"{len(SEMENTES_COMITE)} sementes; score = fração de votos"))

    PASTA_RESULTADOS.mkdir(exist_ok=True)
    figuras = []
    if not args.sem_figuras:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _estilo(plt)
        figura_comparacao([l for l in linhas if l["modelo"] != "Chutar 'ninguém evade'"], teto,
                          PASTA_RESULTADOS / "comparacao_modelos.png", plt)
        figura_sensibilidade(
            [("ARTMAP — versão anterior", linhas[-3]["acuracias"]),
             ("ARTMAP — corrigido (1 rede)", linhas[-2]["acuracias"]),
             (f"Comitê de {N_REDES} redes", linhas[-1]["acuracias"])],
            PASTA_RESULTADOS / "sensibilidade_ordem.png", plt)
        figura_matriz(matriz, PASTA_RESULTADOS / "matriz_confusao.png", plt)
        figura_dobras(acc_dobras, PASTA_RESULTADOS / "estabilidade_dobras.png", plt)
        figuras = ["comparacao_modelos.png", "sensibilidade_ordem.png", "matriz_confusao.png", "estabilidade_dobras.png"]

    print("Treinando o modelo de produção com 100% da base...")
    metricas_salvas = {k: (None if v is None else round(float(v), 4)) for k, v in metricas_producao.items()}
    metricas_salvas.update({"dobras": 5, "n_redes": N_REDES, "metodo": "StratifiedKFold(5, shuffle=True, random_state=42)"})
    servico = treinar_servico(X, y, {"base_sintetica": int(len(y)), "casos_reais": 0, "casos_ignorados": 0},
                              metricas_salvas, n_redes=N_REDES, semente=SEMENTE)
    ARQUIVO_MODELO.parent.mkdir(exist_ok=True)
    servico.salvar_arquivo(ARQUIVO_MODELO)

    vn, fp, fn, vp = matriz.ravel()
    relatorio = f"""# Avaliação do modelo de predição de evasão

Gerado por `treinar_modelo.py` em {datetime.now():%d/%m/%Y %H:%M}. Base: **{len(y)} alunos sintéticos**
({pct(y.mean())} evadiram), validação cruzada estratificada com 5 dobras (`random_state=42`, a mesma
divisão do experimento original do TCC). Versão do modelo publicado: `{servico.versao}`.

> **Limitação principal:** a base é **sintética**, gerada por regras (`gerar_dados_ficticios.py`). No gerador,
> alunos "medianos" evadem 30% das vezes e "anomalias" 50% *ao acaso*, então existe um **teto teórico de
> {pct(teto)}** de acurácia (erro de Bayes) que nenhum modelo supera. Os números abaixo medem o quanto cada
> modelo redescobre as regras do gerador — não o desempenho em alunos reais (ver `validar_base_publica.py`).

{tabela_markdown(linhas)}

**Como ler:**
- **Recall (evadiu)** é a métrica principal: dos alunos que evadiram, quantos o modelo alertou. Não alertar um
  aluno em risco custa mais caro que um alarme falso.
- **±** é o desvio-padrão da acurácia entre execuções que só mudam a ordem de apresentação dos alunos (rede
  única) ou a semente das permutações (comitê). O Fuzzy ARTMAP depende da ordem de treino.
- **Versão anterior** reproduz exatamente a implementação original. Na ordem do CSV ela chega a
  {pct(anterior[0]['acuracia'])} (o número da primeira versão do TCC), mas a média entre ordens é
  {pct(linhas[-3]['acuracia'])}: aquele resultado dependia da ordem dos alunos no arquivo.
- **Comitê**: {N_REDES} redes treinadas em ordens diferentes votam; o score de risco é a fração de votos para
  "evade" (Baixo < 1/3 ≤ Médio < 2/3 ≤ Alto).

## Efeito de cada correção (rede única, {N_ORDENS} ordens de treino)

A implementação original tinha dois desvios em relação ao algoritmo de Carpenter et al. (1992):

1. **Aprendizado antes do map field:** a categoria escolhida tinha os pesos atualizados *antes* de conferir o
   rótulo; quando o rótulo estava errado, ela já tinha absorvido um aluno da outra classe.
2. **Vigilância residual na predição:** `prever()` usava a vigilância que sobrava do match-tracking do último
   aluno do treino e, quando nenhuma categoria passava, respondia "não evade" por padrão.

{tabela_markdown(decomposicao)}

A instabilidade entre ordens (± 5 p.p.) vinha quase toda do defeito 2; o efeito isolado do defeito 1 é pequeno
nesta base, mas a correção alinha a implementação ao algoritmo publicado.

## Matriz de confusão do comitê (semente 42)

| | Previsto: não evade | Previsto: evade |
|---|---|---|
| **Real: não evadiu** | {vn} | {fp} (falsos alarmes) |
| **Real: evadiu** | {fn} (não detectados) | {vp} |

Acurácia por dobra: {", ".join(pct(a) for a in acc_dobras)}.

## Figuras

{chr(10).join(f"![{f}]({f})" for f in figuras) if figuras else "(geradas sem `--sem-figuras`)"}
"""
    (PASTA_RESULTADOS / "avaliacao.md").write_text(relatorio, encoding="utf-8")
    (PASTA_RESULTADOS / "avaliacao.json").write_text(
        json.dumps({"teto_bayes": teto, "linhas": linhas}, ensure_ascii=False, indent=1), encoding="utf-8")

    for l in linhas:
        print(f"  {l['modelo']:<40} acc {pct(l['acuracia']):>6}  recall {pct(l['recall']):>6}")
    print(f"Modelo salvo em {ARQUIVO_MODELO} (versão {servico.versao}). Relatório em {PASTA_RESULTADOS / 'avaliacao.md'}.")


if __name__ == "__main__":
    main()
