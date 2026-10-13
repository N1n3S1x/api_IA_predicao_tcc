"""Utilidades dos testes: caminho de importação e um repositório em memória
que imita o Firestore (mesma interface de repositorio.RepositorioFirestore)."""

import json
import sys
from pathlib import Path

PASTA_IA = Path(__file__).resolve().parents[1]
if str(PASTA_IA) not in sys.path:
    sys.path.insert(0, str(PASTA_IA))

ARQUIVO_MODELO = PASTA_IA / "modelo" / "modelo_fuzzy_artmap.json"


class RepositorioMemoria:
    def __init__(self, alunos=None, usuarios=None):
        self.alunos = {k: dict(v) for k, v in (alunos or {}).items()}
        self.usuarios = dict(usuarios or {})
        self.historico: dict[str, list] = {}
        self.modelos: dict[str, dict] = {}

    def obter_aluno(self, aluno_id):
        doc = self.alunos.get(aluno_id)
        return dict(doc) if doc is not None else None

    def listar_alunos(self, turma=None, ids=None):
        itens = [(i, self.alunos[i]) for i in ids if i in self.alunos] if ids else list(self.alunos.items())
        return [(i, dict(d)) for i, d in itens if not turma or d.get("turma") == turma]

    def listar_desfechos(self):
        return [(i, dict(d)) for i, d in self.alunos.items() if d.get("statusMatricula") in ("Evadido", "Concluído")]

    def salvar_risco(self, aluno_id, campos, historico):
        self.alunos[aluno_id].update(campos)
        self.historico.setdefault(aluno_id, []).append(historico)

    def marcar_desfecho_aprendido(self, aluno_id, info):
        self.alunos[aluno_id]["desfechoAprendido"] = dict(info)

    def obter_perfil_usuario(self, uid):
        return self.usuarios.get(uid)

    def carregar_modelo(self):
        return self.modelos.get("atual")

    def salvar_modelo(self, dados, manter_anterior=False):
        if manter_anterior and "atual" in self.modelos:
            self.modelos["anterior"] = self.modelos["atual"]
        self.modelos["atual"] = json.loads(json.dumps(dados))  # garante que é serializável


def tem_lista_aninhada(valor) -> bool:
    """O Firestore não aceita arrays diretamente dentro de arrays."""
    if isinstance(valor, dict):
        return any(tem_lista_aninhada(v) for v in valor.values())
    if isinstance(valor, list):
        return any(isinstance(v, list) or tem_lista_aninhada(v) for v in valor)
    return False
