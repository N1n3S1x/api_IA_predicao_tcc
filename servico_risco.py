"""Serviço de risco de evasão: junta o comitê Fuzzy ARTMAP, a normalização e
as regras de negócio. Não depende do FastAPI nem do Firestore (fácil de testar).
"""

from __future__ import annotations

import csv
import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import preprocessamento as pp
from fuzzy_artmap import ComiteFuzzyARTMAP

NAO_CALCULADO = "Não calculado"
# Score = fração das redes do comitê que preveem evasão.
LIMITES_NIVEL = ((1 / 3, "Baixo"), (2 / 3, "Médio"))  # a partir de 2/3: "Alto"
NOME_FAIXA = {"baixa": "baixa", "media": "média", "alta": "alta"}


def classificar(score: float) -> str:
    for limite, nome in LIMITES_NIVEL:
        if score < limite:
            return nome
    return "Alto"


def agora_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ResultadoRisco:
    risco: float | None
    classificacao: str
    fatores: list[str] = field(default_factory=list)
    perfil: str | None = None
    motivo: str | None = None
    entrada: dict | None = None
    versao_modelo: str | None = None

    @property
    def calculado(self) -> bool:
        return self.risco is not None

    def para_resposta(self) -> dict:
        return {
            "risco": self.risco,
            "classificacao": self.classificacao,
            "fatores": self.fatores,
            "perfil": self.perfil,
            "motivo": self.motivo,
            "entrada": self.entrada,
            "versaoModelo": self.versao_modelo,
        }

    def campos_aluno(self) -> dict:
        """Campos gravados em `alunos/{id}` (somente a API escreve estes campos)."""
        return {
            "riscoEvasao": self.risco,
            "classificacaoRisco": self.classificacao,
            "fatoresRisco": self.fatores,
            "perfilRisco": self.perfil,
            "motivoRisco": self.motivo,
            "versaoModelo": self.versao_modelo,
        }


def descrever_perfil(explicacao: dict) -> str | None:
    """Traduz o hiper-retângulo da categoria vencedora para linguagem simples."""
    if not explicacao or explicacao.get("inferior") is None:
        return None
    inf, sup = explicacao["inferior"], explicacao["superior"]
    faixas = [nome for nome, v in pp.FAIXAS_RENDA.items() if inf[2] - 1e-9 <= v <= sup[2] + 1e-9]
    if not faixas:
        renda = "qualquer"
    elif len(faixas) == 1:
        renda = NOME_FAIXA[faixas[0]]
    else:
        renda = f"{NOME_FAIXA[faixas[0]]} a {NOME_FAIXA[faixas[-1]]}"
    desfecho = "evadiram" if explicacao["rotulo"] == 1 else "não evadiram"
    texto = (
        f"Frequência entre {inf[0] * 100:.0f}% e {sup[0] * 100:.0f}%, média entre "
        f"{inf[1] * 10:.1f} e {sup[1] * 10:.1f} e renda {renda}: perfil aprendido com "
        f"{explicacao['alunos']} aluno(s) da base que {desfecho}."
    )
    if not explicacao.get("dentro_da_vigilancia"):
        texto += " O aluno não se encaixa bem em nenhum perfil conhecido; foi usado o mais próximo."
    return texto


def _distribuicao(contagem: dict) -> dict:
    total = sum(contagem.values())
    return {f: (n / total if total else 1 / len(contagem)) for f, n in contagem.items()}


def contagem_renda(X: np.ndarray) -> dict:
    return {f: int(np.sum(np.isclose(X[:, 2], v))) for f, v in pp.FAIXAS_RENDA.items()}


def gerar_versao(comite: ComiteFuzzyARTMAP) -> str:
    resumo = hashlib.sha1(json.dumps(comite.para_dict(), sort_keys=True).encode()).hexdigest()[:6]
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M}-{resumo}"


class ServicoRisco:
    def __init__(self, comite: ComiteFuzzyARTMAP, metadados: dict):
        self.comite = comite
        self.metadados = metadados
        self._trava = threading.RLock()

    @property
    def versao(self) -> str | None:
        return self.metadados.get("versao")

    # ---------------- persistência ----------------
    def para_dict(self) -> dict:
        with self._trava:
            return {"modelo": self.comite.para_dict(), "metadados": json.loads(json.dumps(self.metadados))}

    @classmethod
    def de_dict(cls, dados: dict) -> "ServicoRisco":
        return cls(ComiteFuzzyARTMAP.de_dict(dados["modelo"]), dict(dados["metadados"]))

    def salvar_arquivo(self, caminho: Path) -> None:
        Path(caminho).write_text(json.dumps(self.para_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def carregar_arquivo(cls, caminho: Path) -> "ServicoRisco":
        return cls.de_dict(json.loads(Path(caminho).read_text(encoding="utf-8")))

    # ---------------- predição ----------------
    def _score(self, frequencia: float, nota: float, faixa: str | None) -> tuple[float, dict]:
        if faixa is not None:
            x = pp.normalizar_entrada(frequencia, nota, faixa)
            return self.comite.score(x), self.comite.explicar(x)
        # Renda não informada: média do score em cada faixa, ponderada pela
        # distribuição de renda da base de treino (marginalização).
        dist = _distribuicao(self.metadados["contagem_renda"])
        score = sum(p * self.comite.score(pp.normalizar_entrada(frequencia, nota, f)) for f, p in dist.items())
        faixa_comum = max(dist, key=dist.get)
        return score, self.comite.explicar(pp.normalizar_entrada(frequencia, nota, faixa_comum))

    def avaliar(self, frequencia: float, nota_media: float, faixa_renda=None) -> ResultadoRisco:
        pp.validar_indicadores(frequencia, nota_media)
        faixa = pp.normalizar_faixa_renda(faixa_renda)
        with self._trava:
            score, explicacao = self._score(frequencia, nota_media, faixa)
            versao = self.versao
        return ResultadoRisco(
            risco=round(score, 4),
            classificacao=classificar(score),
            fatores=pp.fatores_de_atencao(frequencia, nota_media, faixa),
            perfil=descrever_perfil(explicacao),
            entrada={"frequencia": frequencia, "notaMedia": nota_media, "faixaRenda": faixa},
            versao_modelo=versao,
        )

    def _nao_calculado(self, motivo: str) -> ResultadoRisco:
        return ResultadoRisco(None, NAO_CALCULADO, motivo=motivo, versao_modelo=self.versao)

    def avaliar_documento(self, doc: dict) -> ResultadoRisco:
        """Risco de um documento `alunos/{id}`; nunca assume "baixo" por falta de dados."""
        status = doc.get("statusMatricula")
        if not pp.esta_ativo(status):
            return self._nao_calculado(f"Situação '{status}': o risco só é calculado para alunos ativos.")
        try:
            ind = pp.extrair_indicadores(doc)
        except pp.DadosInvalidos as erro:
            return self._nao_calculado(f"Dados inválidos: {erro}")
        if not ind.completos:
            return self._nao_calculado("Faltam dados para o cálculo: " + ", ".join(ind.faltando) + ".")
        return self.avaliar(ind.frequencia, ind.nota_media, ind.faixa_renda)

    # ---------------- aprendizado ----------------
    @staticmethod
    def caso_de_treino(doc: dict) -> tuple[tuple[np.ndarray, int] | None, str | None]:
        """Converte um aluno com desfecho conhecido em (entrada, rótulo)."""
        rotulo = pp.rotulo_do_status(doc.get("statusMatricula"))
        if rotulo is None:
            return None, "Desfecho desconhecido (aluno ativo ou transferido): não entra no treino."
        try:
            ind = pp.extrair_indicadores(doc)
        except pp.DadosInvalidos as erro:
            return None, f"Dados inválidos: {erro}"
        if not ind.completos:
            return None, "Faltam dados: " + ", ".join(ind.faltando) + "."
        if ind.faixa_renda is None:
            return None, "Faixa de renda não informada: o caso não pode ser usado no treino."
        return (pp.normalizar_entrada(ind.frequencia, ind.nota_media, ind.faixa_renda), rotulo), None

    def aprender_caso(self, x: np.ndarray, rotulo: int) -> None:
        """Aprendizado incremental: todas as redes do comitê aprendem o novo caso."""
        with self._trava:
            self.comite.aprender(x, rotulo)
            faixa = min(pp.FAIXAS_RENDA, key=lambda f: abs(pp.FAIXAS_RENDA[f] - x[2]))
            self.metadados["contagem_renda"][faixa] += 1
            self.metadados["casos_incrementais"] = self.metadados.get("casos_incrementais", 0) + 1
            self.metadados["atualizado_em"] = agora_iso()
            self.metadados["versao"] = gerar_versao(self.comite)


def treinar_servico(X, y, origem_dados: dict, metricas: dict | None = None, n_redes=7, semente=42) -> ServicoRisco:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int).ravel()
    comite = ComiteFuzzyARTMAP(n_redes=n_redes, semente=semente).treinar(X, y)
    metadados = {
        "versao": gerar_versao(comite),
        "treinado_em": agora_iso(),
        "atualizado_em": agora_iso(),
        "origem_dados": origem_dados,
        "metricas_validacao": metricas,
        "contagem_renda": contagem_renda(X),
        "casos_incrementais": 0,
        "limites_classificacao": {"medio": LIMITES_NIVEL[0][0], "alto": LIMITES_NIVEL[1][0]},
    }
    return ServicoRisco(comite, metadados)


# ---------------- avaliação (sem scikit-learn, para rodar também na API) ----------------
def dobras_estratificadas(y, k: int = 5, semente: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(semente)
    y = np.asarray(y)
    dobras = [[] for _ in range(k)]
    for classe in np.unique(y):
        for i, idx in enumerate(rng.permutation(np.flatnonzero(y == classe))):
            dobras[i % k].append(idx)
    return [np.array(sorted(d)) for d in dobras]


def metricas_binarias(y_real, y_previsto) -> dict:
    y_real, y_previsto = np.asarray(y_real), np.asarray(y_previsto)
    vp = int(np.sum((y_real == 1) & (y_previsto == 1)))
    fp = int(np.sum((y_real == 0) & (y_previsto == 1)))
    fn = int(np.sum((y_real == 1) & (y_previsto == 0)))
    precisao = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    return {
        "acuracia": float(np.mean(y_real == y_previsto)),
        "precisao": precisao,
        "recall": recall,
        "f1": 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0,
        "n": int(len(y_real)),
    }


def validacao_cruzada(X, y, k=5, n_redes=7, semente=42) -> dict:
    """Validação cruzada estratificada do comitê; retorna métricas agregadas."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int).ravel()
    dobras = dobras_estratificadas(y, k, semente)
    previsto = np.zeros_like(y)
    for teste in dobras:
        treino = np.setdiff1d(np.arange(len(y)), teste)
        comite = ComiteFuzzyARTMAP(n_redes=n_redes, semente=semente).treinar(X[treino], y[treino])
        previsto[teste] = comite.prever_lote(X[teste])
    return {**metricas_binarias(y, previsto), "dobras": k, "n_redes": n_redes}


def carregar_base_csv(caminho: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lê a base sintética (frequencia, nota_media, renda já em [0, 1]; evasao 0/1)."""
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    X = np.array([[float(l["frequencia"]), float(l["nota_media"]), float(l["renda"])] for l in linhas])
    y = np.array([int(float(l["evasao"])) for l in linhas])
    return X, y
