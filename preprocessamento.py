"""Normalização e regras de negócio dos dados do aluno — ÚNICO lugar do sistema
onde frequência, nota e renda são convertidas para a escala do modelo.

O app envia/grava valores "naturais" (frequência em %, nota de 0 a 10 e a faixa
de renda); a API converte aqui. Assim treino, predição e re-treino usam
exatamente a mesma codificação.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import numpy as np

# Faixas de renda familiar usadas no treino (base sintética: 0,0 / 0,5 / 1,0).
FAIXAS_RENDA = {"baixa": 0.0, "media": 0.5, "alta": 1.0}
DESCRICAO_FAIXA = {
    "baixa": "até 2 salários mínimos",
    "media": "de 2 a 5 salários mínimos",
    "alta": "acima de 5 salários mínimos",
}
# Só usado para converter registros antigos que guardavam a renda em reais.
# Valor de 2025 (Decreto nº 12.342/2024) — atualize se a escola usar outro.
SALARIO_MINIMO = 1518.00
LIMITES_FAIXA_SM = (2, 5)

FREQUENCIA_MINIMA_LDB = 75.0  # Lei 9.394/1996 (LDB), art. 24, VI
MEDIA_APROVACAO = 7.0  # mesmos limites da classe Aluno do app
MEDIA_RECUPERACAO = 5.0

STATUS_ATIVO = "ativo"
# Só desfechos CONHECIDOS viram rótulo de treino. Aluno ativo ainda não tem
# desfecho e transferência não é evasão: ambos ficam fora do treino.
ROTULO_POR_STATUS = {"evadido": 1, "concluido": 0}


class DadosInvalidos(ValueError):
    """Valor fora da faixa esperada (ex.: frequência 85 enviada como 0,85)."""


def _sem_acento(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).strip().lower()


# O campo `perfil` já foi gravado com variações ('direcao', 'Diretor', 'Secretário'...).
SINONIMOS_PERFIL = {
    "direcao": "direcao", "diretor": "direcao", "diretora": "direcao", "diretoria": "direcao",
    "secretaria": "secretaria", "secretario": "secretaria",
    "coordenacao": "coordenacao", "coordenador": "coordenacao", "coordenadora": "coordenacao",
    "professor": "professor", "professora": "professor",
    "aluno": "aluno", "aluna": "aluno",
}


def normalizar_perfil(valor) -> str | None:
    """Forma canônica do perfil ('direcao', 'secretaria', 'coordenacao', 'professor', 'aluno')."""
    return SINONIMOS_PERFIL.get(_sem_acento(valor)) if valor else None


def normalizar_faixa_renda(valor) -> str | None:
    """Converte o que estiver gravado para 'baixa' | 'media' | 'alta' (ou None).

    Aceita a faixa por extenso ('média', 'Baixa'...), registros antigos em reais
    (ex.: 1500.0) e valores já codificados (0, 0,5 ou 1).
    """
    if valor is None or valor == "":
        return None
    if isinstance(valor, str):
        chave = _sem_acento(valor)
        if chave in FAIXAS_RENDA:
            return chave
        raise DadosInvalidos(f"Faixa de renda desconhecida: {valor!r}")
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise DadosInvalidos(f"Renda em formato inválido: {valor!r}")
    valor = float(valor)
    if valor < 0:
        raise DadosInvalidos("A renda não pode ser negativa.")
    if valor <= 1.0:  # já codificado na escala do modelo: aproxima da faixa mais próxima
        return min(FAIXAS_RENDA, key=lambda f: abs(FAIXAS_RENDA[f] - valor))
    salarios = valor / SALARIO_MINIMO  # registro antigo em reais
    if salarios <= LIMITES_FAIXA_SM[0]:
        return "baixa"
    if salarios <= LIMITES_FAIXA_SM[1]:
        return "media"
    return "alta"


def validar_indicadores(frequencia: float, nota_media: float) -> None:
    if not 0.0 <= frequencia <= 100.0:
        raise DadosInvalidos(f"Frequência deve estar entre 0 e 100 (%), recebi {frequencia}.")
    if not 0.0 <= nota_media <= 10.0:
        raise DadosInvalidos(f"Nota média deve estar entre 0 e 10, recebi {nota_media}.")


def normalizar_entrada(frequencia: float, nota_media: float, faixa_renda: str) -> np.ndarray:
    """Vetor de entrada do modelo: [frequência/100, nota/10, faixa de renda]."""
    validar_indicadores(frequencia, nota_media)
    faixa = normalizar_faixa_renda(faixa_renda)
    if faixa is None:
        raise DadosInvalidos("A faixa de renda é obrigatória para montar a entrada.")
    return np.array([frequencia / 100.0, nota_media / 10.0, FAIXAS_RENDA[faixa]])


def frequencia_de_contadores(aulas: int, faltas: int) -> float:
    """Frequência (%) a partir dos contadores de chamada gravados pelo app."""
    if aulas <= 0:
        raise DadosInvalidos("Nenhuma aula registrada.")
    faltas = min(max(faltas, 0), aulas)
    return round(100.0 * (aulas - faltas) / aulas, 2)


def rotulo_do_status(status) -> int | None:
    """1 = evadiu, 0 = concluiu, None = desfecho desconhecido (ativo/transferido)."""
    if not status:
        return None
    return ROTULO_POR_STATUS.get(_sem_acento(status))


def esta_ativo(status) -> bool:
    return not status or _sem_acento(status) == STATUS_ATIVO


@dataclass
class Indicadores:
    frequencia: float | None = None
    nota_media: float | None = None
    faixa_renda: str | None = None
    faltando: list[str] = field(default_factory=list)

    @property
    def completos(self) -> bool:
        return not self.faltando


def _numero(valor) -> float | None:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    return float(valor)


def extrair_indicadores(doc: dict) -> Indicadores:
    """Lê frequência, nota média e faixa de renda de um documento `alunos/{id}`.

    - Frequência: dos contadores de chamada (`aulasRegistradas`/`faltas`) quando
      existem; senão do campo `frequencia` (%) de registros antigos.
    - Nota: `notaMediaGeral` (0 a 10), mantida pelo app ao lançar notas.
    - Renda: `faixaRenda`; registros antigos com `renda` em reais são convertidos.
      Renda não informada NÃO impede o cálculo (o serviço marginaliza a faixa).
    """
    ind = Indicadores()

    aulas = _numero(doc.get("aulasRegistradas"))
    if aulas and aulas > 0:
        ind.frequencia = frequencia_de_contadores(int(aulas), int(_numero(doc.get("faltas")) or 0))
    else:
        ind.frequencia = _numero(doc.get("frequencia"))
    if ind.frequencia is None:
        ind.faltando.append("frequência")

    ind.nota_media = _numero(doc.get("notaMediaGeral"))
    if ind.nota_media is None:
        ind.faltando.append("nota média")

    bruto = doc.get("faixaRenda")
    if bruto in (None, ""):
        bruto = doc.get("renda")
    try:
        ind.faixa_renda = normalizar_faixa_renda(bruto)
    except DadosInvalidos:
        ind.faixa_renda = None

    if ind.frequencia is not None and ind.nota_media is not None:
        validar_indicadores(ind.frequencia, ind.nota_media)
    return ind


def fatores_de_atencao(frequencia: float, nota_media: float, faixa_renda: str | None) -> list[str]:
    """Fatores objetivos, em linguagem simples, para a equipe pedagógica."""
    fatores = []
    if frequencia < FREQUENCIA_MINIMA_LDB:
        fatores.append(
            f"Frequência de {frequencia:.0f}%, abaixo do mínimo legal de {FREQUENCIA_MINIMA_LDB:.0f}% (LDB, art. 24)."
        )
    if nota_media < MEDIA_RECUPERACAO:
        fatores.append(f"Média geral {nota_media:.1f}, abaixo de {MEDIA_RECUPERACAO:.1f} (faixa de recuperação).")
    elif nota_media < MEDIA_APROVACAO:
        fatores.append(f"Média geral {nota_media:.1f}, abaixo da média de aprovação ({MEDIA_APROVACAO:.1f}).")
    if faixa_renda == "baixa":
        fatores.append(f"Contexto socioeconômico: renda familiar {DESCRICAO_FAIXA['baixa']}.")
    elif faixa_renda is None:
        fatores.append("Faixa de renda não informada: o cálculo considerou a distribuição de renda da base.")
    return fatores
