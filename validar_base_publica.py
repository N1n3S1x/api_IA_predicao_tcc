"""Valida o MÉTODO (comitê Fuzzy ARTMAP) numa base PÚBLICA com alunos reais.

Base: "Predict Students' Dropout and Academic Success" (UCI, id 697), 4.424 alunos
de ensino superior em Portugal. Licença CC BY 4.0. Cite como:
  REALINHO, V.; VIEIRA MARTINS, M.; MACHADO, J.; BAPTISTA, L. Predict Students'
  Dropout and Academic Success [Dataset]. UCI Machine Learning Repository, 2021.
  https://doi.org/10.24432/C5MC89

Mesmo princípio do sistema: só entram desfechos conhecidos (Dropout = evadiu,
Graduate = concluiu); "Enrolled" (ainda matriculado) fica de fora.

Uso: python validar_base_publica.py   (baixa a base na 1ª execução)
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fuzzy_artmap import ComiteFuzzyARTMAP
from treinar_modelo import cv_sklearn, metricas, pct, resumir, tabela_markdown

PASTA = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
ARQUIVO = PASTA / "dados" / "publicos" / "uci_697_dropout.csv"
RELATORIO = PASTA / "resultados" / "validacao_base_publica.md"
SEMENTE = 42

# Análogo (não idêntico) aos 3 atributos do TCC: participação, desempenho e
# contexto socioeconômico do 1º semestre.
ANALOGO_TCC = {
    "participacao": "fração das disciplinas do 1º semestre com avaliação (1 - sem avaliação / inscritas)",
    "desempenho": "média das notas do 1º semestre (escala 0 a 20)",
    "socioeconomico": "mensalidades em dia (1) ou atrasadas (0)",
}
# Códigos nominais (curso, nacionalidade, ocupação dos pais...) ficam de fora:
# escalar um código numérico não tem significado.
NUMERICOS = [
    "Previous qualification (grade)", "Admission grade", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International",
    "Daytime/evening attendance",
    *[f"Curricular units {s} sem ({c})" for s in ("1st", "2nd")
      for c in ("credited", "enrolled", "evaluations", "approved", "grade", "without evaluations")],
    "Unemployment rate", "Inflation rate", "GDP",
]


def baixar_base() -> pd.DataFrame:
    if not ARQUIVO.exists():
        print(f"Baixando a base da UCI para {ARQUIVO} ...")
        with urllib.request.urlopen(URL, timeout=120) as resposta:
            conteudo = resposta.read()
        ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
        ARQUIVO.write_bytes(zipfile.ZipFile(io.BytesIO(conteudo)).read("data.csv"))  # só o CSV
    df = pd.read_csv(ARQUIVO, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df


def atributos_analogos(df: pd.DataFrame) -> np.ndarray:
    inscritas = df["Curricular units 1st sem (enrolled)"].replace(0, np.nan)
    participacao = (1 - df["Curricular units 1st sem (without evaluations)"] / inscritas).fillna(0).clip(0, 1)
    desempenho = (df["Curricular units 1st sem (grade)"] / 20).clip(0, 1)
    socioeconomico = df["Tuition fees up to date"].astype(float)
    return np.column_stack([participacao, desempenho, socioeconomico])


def cv_comite(X, y, folds, escalar: bool) -> dict:
    """Comitê com os mesmos hiperparâmetros do TCC; min-max ajustado só no treino."""
    previsto, score = np.zeros_like(y), np.zeros(len(y))
    for treino, teste in folds:
        Xtr, Xte = X[treino], X[teste]
        if escalar:
            minimo, maximo = Xtr.min(axis=0), Xtr.max(axis=0)
            amplitude = np.where(maximo > minimo, maximo - minimo, 1.0)
            Xtr = (Xtr - minimo) / amplitude
            Xte = np.clip((Xte - minimo) / amplitude, 0, 1)
        comite = ComiteFuzzyARTMAP(n_redes=7, semente=SEMENTE, n_entradas=X.shape[1]).treinar(Xtr, y[treino])
        score[teste] = comite.score_lote(Xte)
        previsto[teste] = (score[teste] >= 0.5).astype(int)
    return metricas(y, previsto, score)


def avaliar(X, y, escalar: bool) -> list[dict]:
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE).split(X, y))
    modelos = [
        ("Chutar 'ninguém evade'", DummyClassifier(strategy="most_frequent")),
        ("Regressão logística", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("Árvore de decisão (prof. 3)", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("Random Forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0)),
    ]
    linhas = [resumir(nome, [cv_sklearn(modelo, X, y, folds)]) for nome, modelo in modelos]
    linhas.append(resumir("Comitê de 7 redes Fuzzy ARTMAP", [cv_comite(X, y, folds, escalar)], "hiperparâmetros do TCC"))
    return linhas


def main() -> None:
    df = baixar_base()
    df = df[df["Target"].isin(["Dropout", "Graduate"])].reset_index(drop=True)
    y = (df["Target"] == "Dropout").astype(int).to_numpy()
    print(f"{len(y)} alunos com desfecho conhecido ({y.mean():.1%} evadiram). Avaliando...")

    analogo = avaliar(atributos_analogos(df), y, escalar=False)
    completo = avaliar(df[NUMERICOS].to_numpy(dtype=float), y, escalar=True)

    descricao = "\n".join(f"- **{k}**: {v}" for k, v in ANALOGO_TCC.items())
    RELATORIO.parent.mkdir(exist_ok=True)
    RELATORIO.write_text(
        f"""# Validação do método numa base pública com alunos reais

Gerado por `validar_base_publica.py` em {datetime.now():%d/%m/%Y %H:%M}.

**Base:** *Predict Students' Dropout and Academic Success* (UCI Machine Learning Repository, id 697, licença
CC BY 4.0) — REALINHO, V.; VIEIRA MARTINS, M.; MACHADO, J.; BAPTISTA, L., 2021,
https://doi.org/10.24432/C5MC89. Alunos de **ensino superior em Portugal**; foram mantidos só os
{len(y)} com desfecho conhecido (*Dropout* = evadiu, *Graduate* = concluiu; {pct(y.mean())} evadiram).
Validação cruzada estratificada, 5 dobras, `random_state=42`.

> Esta base **não** é do mesmo contexto do sistema (educação básica no Brasil) e não tem as mesmas variáveis.
> Ela serve para mostrar que o *método* funciona com dados reais e para compará-lo com outros modelos.

## 1. Três atributos análogos aos do TCC

{descricao}

{tabela_markdown(analogo)}

## 2. Todos os atributos numéricos ({len(NUMERICOS)})

Inclui desempenho e aprovações dos dois semestres, idade, bolsa, dívidas e indicadores econômicos.
Códigos nominais (curso, nacionalidade, ocupação dos pais) ficaram de fora.

{tabela_markdown(completo)}
""",
        encoding="utf-8",
    )
    for titulo, linhas in (("3 atributos análogos", analogo), ("atributos numéricos", completo)):
        print(f"--- {titulo} ---")
        for l in linhas:
            print(f"  {l['modelo']:<34} acc {pct(l['acuracia']):>6}  recall {pct(l['recall']):>6}  F1 {l['f1']:.3f}")
    print(f"Relatório: {RELATORIO}")


if __name__ == "__main__":
    main()
