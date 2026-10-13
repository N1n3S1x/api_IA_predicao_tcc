"""Cria (ou atualiza) as contas de teste e, opcionalmente, alunos de demonstração.

Com as regras do Firestore (firestore.rules), ninguém consegue dar a si mesmo um
perfil de equipe (professor, secretaria, direção...). Por isso as contas de teste
são criadas aqui, com a chave de serviço (Admin SDK), e não pela tela de teste do
app. São as mesmas contas dos cartões de "Preenchimento rápido" do login.

Uso (na pasta ia/, com chave_firebase.json ou FIREBASE_CREDENTIALS_JSON):
  python criar_usuarios_teste.py                        contas de teste
  python criar_usuarios_teste.py --alunos-demo 40       + 40 alunos fictícios com notas e frequência
  python criar_usuarios_teste.py --remover-demo         apaga os alunos de demonstração
  python criar_usuarios_teste.py --sincronizar-perfis   grava o perfil de TODOS os usuários como
                                                        custom claim (rode uma vez após publicar as regras)

Depois de mudar claims, o usuário precisa sair e entrar de novo no app.
"""

from __future__ import annotations

import argparse
import random
import sys

import gerar_dados_ficticios
from preprocessamento import normalizar_perfil
from repositorio import conectar_firestore
from servico_risco import ServicoRisco

SENHA_TESTE = "123456"
TURMAS = ["9º Ano A", "9º Ano B"]
DISCIPLINAS = ["Matemática", "Português", "Ciências", "História"]
FAIXA_POR_VALOR = {0.0: "baixa", 0.5: "media", 1.0: "alta"}

CONTAS = [
    {"email": "joao@aluno.com", "nome": "João Aluno", "perfil": "aluno", "matricula": 2026001},
    {"email": "maria@professor.com", "nome": "Maria Professora", "perfil": "professor", "matricula": 2026002},
    {"email": "carlos@direcao.com", "nome": "Carlos Diretor", "perfil": "direcao", "matricula": 2026003},
    {"email": "ana@coordenacao.com", "nome": "Ana Coordenadora", "perfil": "coordenacao", "matricula": 2026005},
    {"email": "pedro@secretaria.com", "nome": "Pedro Secretário", "perfil": "secretaria", "matricula": 2026004},
]


def obter_ou_criar_conta(auth, conta: dict) -> str:
    try:
        usuario = auth.get_user_by_email(conta["email"])
        auth.update_user(usuario.uid, password=SENHA_TESTE, display_name=conta["nome"])
        return usuario.uid
    except auth.UserNotFoundError:
        return auth.create_user(email=conta["email"], password=SENHA_TESTE, display_name=conta["nome"]).uid


def documento_aluno(nome, matricula, turma, frequencia, nota, faixa, rng: random.Random) -> tuple[dict, dict]:
    """Documento `alunos/{id}` no formato que o app grava + notas por disciplina.
    Só os dois primeiros bimestres são lançados (0,0 = nota ainda não lançada)."""
    notas = {}
    for disciplina in DISCIPLINAS:
        bimestres = [round(min(10.0, max(0.5, nota + rng.uniform(-0.8, 0.8))), 1) for _ in range(2)]
        notas[disciplina] = bimestres + [0.0, 0.0]
    lancadas = [n for lista in notas.values() for n in lista if n > 0]
    aulas = 40
    faltas = round(aulas * (1 - frequencia))
    doc = {
        "nome": nome,
        "matricula": matricula,
        "turma": turma,
        "statusMatricula": "Ativo",
        "faixaRenda": faixa,
        "aulasRegistradas": aulas,
        "faltas": faltas,
        "faltasJustificadas": 0,
        "frequencia": round(100 * (aulas - faltas) / aulas, 2),
        "notaMediaGeral": round(sum(lancadas) / len(lancadas), 2),
    }
    return doc, notas


def gravar_aluno(db, aluno_id: str, doc: dict, notas: dict, servico: ServicoRisco) -> str:
    from firebase_admin import firestore

    ref = db.collection("alunos").document(aluno_id)
    ref.set({**doc, "criadoEm": firestore.SERVER_TIMESTAMP}, merge=True)
    for disciplina, lista in notas.items():
        ref.collection("disciplinas").document(disciplina).set({"nome": disciplina, "notas": lista})
    resultado = servico.avaliar_documento(doc)  # mesmo cálculo da API
    ref.update({**resultado.campos_aluno(), "riscoAtualizadoEm": firestore.SERVER_TIMESTAMP})
    return resultado.classificacao


def sincronizar_perfis(db, auth) -> None:
    """Copia o perfil de DBSistemaTCC para o custom claim `perfil` de cada conta e
    corrige variantes antigas no documento ('Diretor' -> 'direcao')."""
    contagem = {"sincronizados": 0, "corrigidos": 0, "sem_perfil_valido": 0, "sem_conta": 0}
    for snap in db.collection("DBSistemaTCC").stream():
        bruto = (snap.to_dict() or {}).get("perfil")
        perfil = normalizar_perfil(bruto)
        if perfil is None:
            contagem["sem_perfil_valido"] += 1
            print(f"  ! {snap.id}: perfil {bruto!r} não reconhecido (corrija no console)")
            continue
        try:
            atuais = auth.get_user(snap.id).custom_claims or {}
            auth.set_custom_user_claims(snap.id, {**atuais, "perfil": perfil})
        except auth.UserNotFoundError:
            contagem["sem_conta"] += 1
            continue
        if bruto != perfil:
            snap.reference.update({"perfil": perfil})
            contagem["corrigidos"] += 1
        contagem["sincronizados"] += 1
    print(f"Perfis: {contagem}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Contas de teste e alunos de demonstração.")
    parser.add_argument("--alunos-demo", type=int, default=0, help="quantos alunos fictícios criar")
    parser.add_argument("--remover-demo", action="store_true", help="apaga os alunos de demonstração")
    parser.add_argument("--sincronizar-perfis", action="store_true", help="perfil de todos -> custom claim")
    args = parser.parse_args()

    repositorio = conectar_firestore()
    if repositorio is None:
        sys.exit("Sem acesso ao Firebase: coloque a chave em ia/chave_firebase.json.")
    from firebase_admin import auth, firestore

    db = repositorio.db

    if args.sincronizar_perfis:
        sincronizar_perfis(db, auth)
        return

    if args.remover_demo:
        demos = list(db.collection("alunos").where(filter=firestore.FieldFilter("demo", "==", True)).stream())
        for snap in demos:
            db.recursive_delete(snap.reference)
        print(f"{len(demos)} aluno(s) de demonstração removido(s).")
        return

    servico = ServicoRisco.carregar_arquivo(gerar_dados_ficticios.PASTA / "modelo" / "modelo_fuzzy_artmap.json")
    rng = random.Random(7)
    for conta in CONTAS:
        uid = obter_ou_criar_conta(auth, conta)
        auth.set_custom_user_claims(uid, {"perfil": conta["perfil"]})  # lido pelas regras e pela API
        base = {"nome": conta["nome"], "email": conta["email"], "matricula": conta["matricula"]}
        db.collection("DBSistemaTCC").document(uid).set(
            {**base, "perfil": conta["perfil"], "criadoEm": firestore.SERVER_TIMESTAMP}, merge=True
        )
        if conta["perfil"] == "aluno":
            doc, notas = documento_aluno(conta["nome"], conta["matricula"], TURMAS[0], 0.92, 8.0, "media", rng)
            gravar_aluno(db, uid, {**doc, "email": conta["email"]}, notas, servico)
        elif conta["perfil"] == "professor":
            db.collection("professores").document(uid).set(
                {**base, "telefone": "(00) 0000-0000", "disciplinas": DISCIPLINAS[:2], "turmas": TURMAS}, merge=True
            )
        else:
            db.collection("funcionarios").document(uid).set(
                {**base, "telefone": "(00) 0000-0000", "perfil": conta["perfil"]}, merge=True
            )
        print(f"OK  {conta['perfil']:<12} {conta['email']}  (senha {SENHA_TESTE})")

    if args.alunos_demo:
        contagem: dict[str, int] = {}
        for i, (frequencia, nota, renda, _) in enumerate(gerar_dados_ficticios.gerar(args.alunos_demo, semente=7)):
            doc, notas = documento_aluno(
                f"Aluno Demonstração {i + 1:02d}", 9000 + i, TURMAS[i % len(TURMAS)],
                frequencia, nota * 10, FAIXA_POR_VALOR[renda], rng,
            )
            classificacao = gravar_aluno(db, f"demo-{i + 1:03d}", {**doc, "demo": True}, notas, servico)
            contagem[classificacao] = contagem.get(classificacao, 0) + 1
        print(f"{args.alunos_demo} alunos de demonstração criados: {contagem}")


if __name__ == "__main__":
    main()
