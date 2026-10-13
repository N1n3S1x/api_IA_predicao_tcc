"""Gera a base SINTÉTICA de alunos usada no treino (dados/dados_alunos_tcc.csv).

Mesma lógica e mesma semente da versão original (criarDadosFicticios.py): o
arquivo gerado é idêntico. Os perfis agora ficam numa tabela (PERFIS), o que
permite calcular o teto teórico de acurácia da base (erro de Bayes): como o
gerador sorteia o desfecho de parte dos alunos, nenhum modelo acerta 100%.

Uso: python gerar_dados_ficticios.py [--saida dados/dados_alunos_tcc.csv] [--qtd 1500]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
ARQUIVO_PADRAO = PASTA / "dados" / "dados_alunos_tcc.csv"

# nome: (peso, faixa de frequência, faixa de nota, rendas possíveis, P(evadir))
PERFIS = {
    "excelente": (0.4, (0.75, 1.0), (0.70, 1.0), [0.5, 1.0], 0.0),
    "vulneravel": (0.3, (0.30, 0.70), (0.20, 0.60), [0.0, 0.5], 1.0),
    "mediano": (0.2, (0.65, 0.85), (0.50, 0.75), [0.0, 0.5, 1.0], 0.3),  # "corda bamba"
    "anomalia": (0.1, (0.80, 1.0), (0.10, 0.40), [0.0], 0.5),  # força o aprendizado
}


def gerar(qtd_alunos: int = 1500, semente: int = 42) -> list[list[float]]:
    np.random.seed(semente)
    random.seed(semente)
    nomes = list(PERFIS)
    pesos = [PERFIS[n][0] for n in nomes]
    dados = []
    for _ in range(qtd_alunos):
        perfil = random.choices(nomes, weights=pesos)[0]
        _, faixa_freq, faixa_nota, rendas, p_evadir = PERFIS[perfil]
        frequencia = round(np.random.uniform(*faixa_freq), 2)
        nota = round(np.random.uniform(*faixa_nota), 2)
        # Mesma sequência de sorteios do script original (necessária para
        # reproduzir o CSV): sem sorteio quando só há uma opção.
        renda = rendas[0] if len(rendas) == 1 else random.choice(rendas)
        if p_evadir in (0.0, 1.0):
            evasao = p_evadir
        else:
            evasao = np.random.choice([0.0, 1.0], p=[round(1 - p_evadir, 10), p_evadir])
        dados.append([frequencia, nota, renda, evasao])
    return dados


def salvar_csv(dados: list[list[float]], caminho: Path) -> None:
    import pandas as pd

    caminho.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(dados, columns=["frequencia", "nota_media", "renda", "evasao"])
    df.to_csv(caminho, index=False)


def teto_bayes() -> float:
    """Acurácia máxima esperada de QUALQUER modelo nesta base (erro de Bayes).

    Calcula P(x, evadiu) exata em cada célula da grade (frequência e nota com
    2 casas decimais x 3 rendas) e soma o maior entre P(x, 0) e P(x, 1).
    """
    grade = np.round(np.arange(0, 101) / 100, 2)

    def prob_arredondada(a: float, b: float) -> np.ndarray:  # P(round(U(a, b), 2) = v)
        inferior, superior = grade - 0.005, grade + 0.005
        return np.clip(np.minimum(superior, b) - np.maximum(inferior, a), 0, None) / (b - a)

    rendas = [0.0, 0.5, 1.0]
    p_evadiu = np.zeros((101, 101, 3))
    p_ficou = np.zeros((101, 101, 3))
    for peso, faixa_freq, faixa_nota, opcoes, p_evadir in PERFIS.values():
        conjunta = peso * np.outer(prob_arredondada(*faixa_freq), prob_arredondada(*faixa_nota))
        for k, renda in enumerate(rendas):
            if renda in opcoes:
                p_evadiu[:, :, k] += conjunta / len(opcoes) * p_evadir
                p_ficou[:, :, k] += conjunta / len(opcoes) * (1 - p_evadir)
    return float(np.maximum(p_evadiu, p_ficou).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--saida", type=Path, default=ARQUIVO_PADRAO)
    parser.add_argument("--qtd", type=int, default=1500)
    args = parser.parse_args()

    dados = gerar(args.qtd)
    salvar_csv(dados, args.saida)
    evadiram = sum(1 for d in dados if d[3] == 1.0)
    print(f"Base com {args.qtd} alunos gerada em '{args.saida}'.")
    print(f"Evadiram: {evadiram} ({evadiram / args.qtd:.1%}) | Não evadiram: {args.qtd - evadiram}")
    print(f"Teto teórico de acurácia desta base (erro de Bayes): {teto_bayes():.1%}")


if __name__ == "__main__":
    main()
