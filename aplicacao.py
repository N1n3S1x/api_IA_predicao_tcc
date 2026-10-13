"""API do Sistema de Apoio à Decisão: risco de evasão com Fuzzy ARTMAP.

A API é a única responsável por calcular e gravar o risco dos alunos: o app só
pede o cálculo. Assim a normalização fica num lugar só e o cliente não consegue
forjar o risco de ninguém.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import preprocessamento as pp
from servico_risco import ServicoRisco, agora_iso, carregar_base_csv, treinar_servico, validacao_cruzada

log = logging.getLogger("api_ia")
PASTA = Path(__file__).resolve().parent

PERFIS_GESTAO = frozenset({"direcao", "coordenacao", "secretaria"})
PERFIS_EQUIPE = PERFIS_GESTAO | {"professor"}
ORIGENS_PADRAO = "https://sistematcc-4a365.web.app,https://sistematcc-4a365.firebaseapp.com"


def _ligado(valor: str | None, padrao: bool) -> bool:
    if valor is None:
        return padrao
    return valor.strip().lower() not in {"0", "false", "nao", "não", "no", "off"}


@dataclass
class Configuracao:
    exigir_autenticacao: bool = True
    origens_permitidas: list[str] = field(default_factory=lambda: ORIGENS_PADRAO.split(","))
    arquivo_modelo: Path = PASTA / "modelo" / "modelo_fuzzy_artmap.json"
    arquivo_base: Path = PASTA / "dados" / "dados_alunos_tcc.csv"
    # Critério mínimo para publicar um modelo re-treinado (senão mantém o atual)
    recall_minimo: float = 0.70
    f1_minimo: float = 0.70

    @classmethod
    def do_ambiente(cls) -> "Configuracao":
        origens = os.environ.get("ORIGENS_PERMITIDAS", ORIGENS_PADRAO)
        return cls(
            exigir_autenticacao=_ligado(os.environ.get("EXIGIR_AUTENTICACAO"), True),
            origens_permitidas=[o.strip() for o in origens.split(",") if o.strip()],
            recall_minimo=float(os.environ.get("RECALL_MINIMO", 0.70)),
            f1_minimo=float(os.environ.get("F1_MINIMO", 0.70)),
        )


@dataclass
class Usuario:
    uid: str
    perfil: str | None


class EntradaRisco(BaseModel):
    frequencia: float = Field(ge=0, le=100, description="Frequência do aluno em % (0 a 100).")
    nota_media: float = Field(ge=0, le=10, description="Média geral, de 0 a 10.")
    faixa_renda: Literal["baixa", "media", "alta"] | None = Field(
        default=None, description="Faixa de renda familiar (opcional)."
    )


class PedidoRecalculo(BaseModel):
    aluno_ids: list[str] | None = Field(default=None, max_length=500)
    turma: str | None = None


def carregar_servico(repositorio, config: Configuracao) -> ServicoRisco:
    """Modelo publicado no Firestore; se não houver, o arquivo do repositório."""
    if repositorio is not None:
        try:
            dados = repositorio.carregar_modelo()
            if dados:
                servico = ServicoRisco.de_dict(dados)
                log.info("Modelo carregado do Firestore (versão %s).", servico.versao)
                return servico
        except Exception as erro:
            log.error("Modelo do Firestore inválido; usando o arquivo local: %s", erro)
    servico = ServicoRisco.carregar_arquivo(config.arquivo_modelo)
    log.info("Modelo carregado de %s (versão %s).", config.arquivo_modelo.name, servico.versao)
    return servico


def _verificar_token_firebase(token: str) -> dict:
    from firebase_admin import auth

    return auth.verify_id_token(token)


def criar_app(
    servico: ServicoRisco | None = None,
    repositorio=None,
    config: Configuracao | None = None,
    verificar_token=_verificar_token_firebase,
    conectar_banco: bool = True,
) -> FastAPI:
    config = config or Configuracao.do_ambiente()
    if repositorio is None and conectar_banco:
        from repositorio import conectar_firestore

        repositorio = conectar_firestore()

    app = FastAPI(
        title="API Fuzzy ARTMAP - Predição de Evasão",
        description="API do TCC: calcula o risco de evasão escolar e explica o resultado.",
        version="2.0.0",
    )
    app.state.servico = servico or carregar_servico(repositorio, config)
    app.state.retreino = {"estado": "nunca executado"}
    app.state.thread_retreino = None
    trava_retreino = threading.Lock()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.origens_permitidas,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",  # flutter run (porta aleatória)
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.exception_handler(pp.DadosInvalidos)
    def _dados_invalidos(_: Request, erro: pp.DadosInvalidos):
        return JSONResponse(status_code=422, content={"detail": str(erro)})

    # ---------------- autenticação e perfis ----------------
    def exigir_repositorio():
        if repositorio is None:
            raise HTTPException(503, "Banco de dados indisponível na API.")
        return repositorio

    def usuario_atual(authorization: str | None = Header(default=None)) -> Usuario:
        if not config.exigir_autenticacao:  # só para desenvolvimento local
            return Usuario(uid="desenvolvimento", perfil="direcao")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Envie o token do Firebase no cabeçalho 'Authorization: Bearer <token>'.")
        try:
            dados_token = verificar_token(authorization.split(" ", 1)[1].strip())
        except Exception:
            raise HTTPException(401, "Token inválido ou expirado. Faça login novamente.")
        # Custom claim (definido pelo Admin SDK) evita uma leitura no banco
        perfil = pp.normalizar_perfil(dados_token.get("perfil"))
        if perfil is None:
            perfil = pp.normalizar_perfil(exigir_repositorio().obter_perfil_usuario(dados_token["uid"]))
        return Usuario(uid=dados_token["uid"], perfil=perfil)

    def exigir_perfil(permitidos: frozenset):
        def dependencia(usuario: Usuario = Depends(usuario_atual)) -> Usuario:
            if usuario.perfil not in permitidos:
                raise HTTPException(403, "Seu perfil não tem permissão para esta operação.")
            return usuario

        return dependencia

    equipe = exigir_perfil(PERFIS_EQUIPE)
    gestao = exigir_perfil(PERFIS_GESTAO)
    direcao = exigir_perfil(frozenset({"direcao"}))

    def recalcular(repo, aluno_id: str, doc: dict, usuario: Usuario):
        resultado = app.state.servico.avaliar_documento(doc)
        historico = {**resultado.campos_aluno(), "entrada": resultado.entrada, "solicitadoPor": usuario.uid}
        repo.salvar_risco(aluno_id, resultado.campos_aluno(), historico)
        return resultado

    # ---------------- rotas ----------------
    @app.get("/")
    def status_api():
        """Status da API (também usado pelo app para "acordar" o Render)."""
        return {
            "status": "online",
            "sistema": "API Fuzzy ARTMAP TCC",
            "versaoModelo": app.state.servico.versao,
            "bancoDeDados": repositorio is not None,
        }

    @app.post("/prever-evasao")
    def prever_evasao(entrada: EntradaRisco, usuario: Usuario = Depends(equipe)):
        """Simulação: risco para valores informados (não grava nada no banco)."""
        resultado = app.state.servico.avaliar(entrada.frequencia, entrada.nota_media, entrada.faixa_renda)
        return resultado.para_resposta()

    @app.post("/alunos/{aluno_id}/risco")
    def recalcular_aluno(aluno_id: str, usuario: Usuario = Depends(equipe)):
        """Calcula o risco a partir do documento do aluno e grava o resultado."""
        repo = exigir_repositorio()
        doc = repo.obter_aluno(aluno_id)
        if doc is None:
            raise HTTPException(404, "Aluno não encontrado.")
        return {"alunoId": aluno_id, **recalcular(repo, aluno_id, doc, usuario).para_resposta()}

    @app.post("/alunos/risco/recalcular")
    def recalcular_varios(pedido: PedidoRecalculo, usuario: Usuario = Depends(equipe)):
        """Recalcula vários alunos (lista de ids, uma turma ou, para a gestão, todos)."""
        repo = exigir_repositorio()
        if not pedido.aluno_ids and not pedido.turma and usuario.perfil not in PERFIS_GESTAO:
            raise HTTPException(403, "Só a gestão pode recalcular todos os alunos de uma vez.")
        por_classificacao: dict[str, int] = {}
        for aluno_id, doc in repo.listar_alunos(turma=pedido.turma, ids=pedido.aluno_ids):
            classificacao = recalcular(repo, aluno_id, doc, usuario).classificacao
            por_classificacao[classificacao] = por_classificacao.get(classificacao, 0) + 1
        return {
            "processados": sum(por_classificacao.values()),
            "porClassificacao": por_classificacao,
            "versaoModelo": app.state.servico.versao,
        }

    @app.post("/alunos/{aluno_id}/desfecho")
    def registrar_desfecho(aluno_id: str, usuario: Usuario = Depends(gestao)):
        """Chamado quando a situação do aluno muda. Se o desfecho for conhecido
        (Evadido/Concluído), o modelo aprende o caso na hora (aprendizado
        incremental do ARTMAP). Em todos os casos o risco é atualizado."""
        repo = exigir_repositorio()
        doc = repo.obter_aluno(aluno_id)
        if doc is None:
            raise HTTPException(404, "Aluno não encontrado.")
        servico = app.state.servico
        caso, mensagem = servico.caso_de_treino(doc)
        aprendido = False
        if caso is not None:
            x, rotulo = caso
            anterior = doc.get("desfechoAprendido") or {}
            if anterior.get("rotulo") == rotulo:
                mensagem = "Este desfecho já tinha sido aprendido pelo modelo."
            elif "rotulo" in anterior:
                mensagem = "O desfecho mudou depois de aprendido; rode o re-treino completo para corrigir o modelo."
            else:
                servico.aprender_caso(x, rotulo)
                repo.salvar_modelo(servico.para_dict())
                repo.marcar_desfecho_aprendido(aluno_id, {"rotulo": rotulo, "versaoModelo": servico.versao})
                aprendido = True
                mensagem = "Desfecho aprendido pelo modelo (aprendizado incremental)."
        resultado = recalcular(repo, aluno_id, doc, usuario)
        return {
            "alunoId": aluno_id,
            "aprendido": aprendido,
            "mensagem": mensagem,
            "versaoModelo": servico.versao,
            "risco": resultado.para_resposta(),
        }

    @app.get("/modelo")
    def info_modelo(usuario: Usuario = Depends(gestao)):
        servico = app.state.servico
        meta = servico.metadados
        return {
            "versao": servico.versao,
            "treinadoEm": meta.get("treinado_em"),
            "atualizadoEm": meta.get("atualizado_em"),
            "origemDados": meta.get("origem_dados"),
            "metricasValidacao": meta.get("metricas_validacao"),
            "casosIncrementais": meta.get("casos_incrementais", 0),
            "categoriasPorRede": servico.comite.n_categorias,
            "retreino": app.state.retreino,
        }

    def executar_retreino(repo):
        try:
            X_base, y_base = carregar_base_csv(config.arquivo_base)
            X_reais, y_reais, ignorados = [], [], 0
            for _, doc in repo.listar_desfechos():
                caso, _ = ServicoRisco.caso_de_treino(doc)
                if caso is None:
                    ignorados += 1
                    continue
                X_reais.append(caso[0])
                y_reais.append(caso[1])
            X = np.vstack([X_base, np.array(X_reais)]) if X_reais else X_base
            y = np.concatenate([y_base, np.array(y_reais, dtype=int)])
            atual = app.state.servico
            metricas = validacao_cruzada(X, y, n_redes=atual.comite.n_redes)
            origem = {"base_sintetica": int(len(y_base)), "casos_reais": len(y_reais), "casos_ignorados": ignorados}
            resumo = {
                "metricas": metricas,
                "origemDados": origem,
                "metricasModeloAnterior": atual.metadados.get("metricas_validacao"),
            }
            if metricas["recall"] < config.recall_minimo or metricas["f1"] < config.f1_minimo:
                app.state.retreino = {
                    "estado": "rejeitado",
                    "concluidoEm": agora_iso(),
                    "mensagem": (
                        f"O modelo novo ficou abaixo do mínimo (recall ≥ {config.recall_minimo:.0%} e "
                        f"F1 ≥ {config.f1_minimo:.0%}); o modelo atual foi mantido."
                    ),
                    **resumo,
                }
                return
            novo = treinar_servico(X, y, origem, metricas, n_redes=atual.comite.n_redes)
            repo.salvar_modelo(novo.para_dict(), manter_anterior=True)
            app.state.servico = novo
            app.state.retreino = {
                "estado": "concluido",
                "concluidoEm": agora_iso(),
                "versao": novo.versao,
                "mensagem": "Modelo re-treinado e publicado. Recalcule os riscos para atualizar os alunos.",
                **resumo,
            }
        except Exception as erro:
            log.exception("Falha no re-treino")
            app.state.retreino = {"estado": "erro", "concluidoEm": agora_iso(), "mensagem": str(erro)}
        finally:
            trava_retreino.release()

    @app.post("/modelo/retreinar", status_code=202)
    def retreinar(usuario: Usuario = Depends(direcao)):
        """Re-treino completo (base + desfechos confirmados) em segundo plano.
        O modelo novo só é publicado se passar no critério mínimo de qualidade."""
        repo = exigir_repositorio()
        if not trava_retreino.acquire(blocking=False):
            raise HTTPException(409, "Já existe um re-treino em andamento.")
        app.state.retreino = {"estado": "executando", "iniciadoEm": agora_iso(), "solicitadoPor": usuario.uid}
        app.state.thread_retreino = threading.Thread(target=executar_retreino, args=(repo,), daemon=True)
        app.state.thread_retreino.start()
        return app.state.retreino

    return app
