"""Acesso ao Firestore pela API (Firebase Admin SDK, que não passa pelas regras
de segurança). Só a API grava os campos de risco e o modelo treinado.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("api_ia")

PASTA = Path(__file__).resolve().parent
COLECAO_ALUNOS = "alunos"
COLECAO_USUARIOS = "DBSistemaTCC"
COLECAO_MODELOS = "modelos_ia"
STATUS_COM_DESFECHO = ["Evadido", "Concluído"]


class RepositorioFirestore:
    def __init__(self, db):
        from firebase_admin import firestore

        self.db = db
        self._carimbo = firestore.SERVER_TIMESTAMP
        self._filtro = firestore.FieldFilter

    def _alunos(self):
        return self.db.collection(COLECAO_ALUNOS)

    def obter_aluno(self, aluno_id: str) -> dict | None:
        snap = self._alunos().document(aluno_id).get()
        return snap.to_dict() if snap.exists else None

    def listar_alunos(self, turma: str | None = None, ids: list[str] | None = None) -> list[tuple[str, dict]]:
        if ids:
            snaps = self.db.get_all([self._alunos().document(i) for i in ids])
        else:
            consulta = self._alunos()
            if turma:
                consulta = consulta.where(filter=self._filtro("turma", "==", turma))
            snaps = consulta.stream()
        return [(s.id, s.to_dict()) for s in snaps if s.exists]

    def listar_desfechos(self) -> list[tuple[str, dict]]:
        consulta = self._alunos().where(filter=self._filtro("statusMatricula", "in", STATUS_COM_DESFECHO))
        return [(s.id, s.to_dict()) for s in consulta.stream()]

    def salvar_risco(self, aluno_id: str, campos: dict, historico: dict) -> None:
        """Atualiza o risco do aluno e acrescenta uma entrada no histórico (auditoria)."""
        ref = self._alunos().document(aluno_id)
        lote = self.db.batch()
        lote.update(ref, {**campos, "riscoAtualizadoEm": self._carimbo})
        lote.set(ref.collection("historicoRisco").document(), {**historico, "em": self._carimbo})
        lote.commit()

    def marcar_desfecho_aprendido(self, aluno_id: str, info: dict) -> None:
        self._alunos().document(aluno_id).update({"desfechoAprendido": {**info, "em": self._carimbo}})

    def obter_perfil_usuario(self, uid: str) -> str | None:
        snap = self.db.collection(COLECAO_USUARIOS).document(uid).get()
        return (snap.to_dict() or {}).get("perfil") if snap.exists else None

    def carregar_modelo(self) -> dict | None:
        snap = self.db.collection(COLECAO_MODELOS).document("atual").get()
        return snap.to_dict() if snap.exists else None

    def salvar_modelo(self, dados: dict, manter_anterior: bool = False) -> None:
        """O disco do Render é apagado a cada reinício: o modelo vive no Firestore."""
        colecao = self.db.collection(COLECAO_MODELOS)
        if manter_anterior:
            atual = colecao.document("atual").get()
            if atual.exists:
                colecao.document("anterior").set(atual.to_dict())
        colecao.document("atual").set(dados)


def conectar_firestore() -> RepositorioFirestore | None:
    """Usa FIREBASE_CREDENTIALS_JSON (Render) ou ia/chave_firebase.json (local)."""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            json_ambiente = os.environ.get("FIREBASE_CREDENTIALS_JSON")
            if json_ambiente:
                credencial = credentials.Certificate(json.loads(json_ambiente))
            else:
                credencial = credentials.Certificate(str(PASTA / "chave_firebase.json"))
            firebase_admin.initialize_app(credencial)
        log.info("Conectado ao Firestore.")
        return RepositorioFirestore(firestore.client())
    except Exception as erro:  # sem credencial: a API sobe, mas sem banco
        log.error("Firestore indisponível: %s", erro)
        return None
