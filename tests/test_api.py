import unittest

from fastapi.testclient import TestClient

import apoio
from aplicacao import Configuracao, criar_app
from servico_risco import ServicoRisco

TOKENS = {"t-dir": "u-dir", "t-sec": "u-sec", "t-prof": "u-prof", "t-aluno": "u-aluno"}
# Variantes reais do campo `perfil` (a tela de teste chegou a gravar "Diretor")
PERFIS = {"u-dir": "Diretor", "u-sec": "secretaria", "u-prof": "professor", "u-aluno": "aluno"}


def verificar_token_falso(token):
    if token == "t-claim":  # perfil no custom claim, sem documento em DBSistemaTCC
        return {"uid": "u-claim", "perfil": "professor"}
    if token not in TOKENS:
        raise ValueError("token inválido")
    return {"uid": TOKENS[token]}


def cabecalho(token):
    return {"Authorization": f"Bearer {token}"}


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.repo = apoio.RepositorioMemoria(
            alunos={
                "a1": {"nome": "Ana", "turma": "9º Ano A", "statusMatricula": "Ativo",
                       "aulasRegistradas": 40, "faltas": 22, "notaMediaGeral": 3.0, "faixaRenda": "baixa"},
                "a2": {"nome": "Bia", "turma": "9º Ano A", "statusMatricula": "Ativo", "faixaRenda": "media"},
                "a3": {"nome": "Caio", "turma": "8º Ano B", "statusMatricula": "Evadido",
                       "frequencia": 45.0, "notaMediaGeral": 3.5, "faixaRenda": "baixa"},
                "a4": {"nome": "Duda", "turma": "8º Ano B", "statusMatricula": "Transferido",
                       "frequencia": 90.0, "notaMediaGeral": 8.0, "faixaRenda": "alta"},
            },
            usuarios=PERFIS,
        )
        self.config = Configuracao(exigir_autenticacao=True)
        self.app = criar_app(
            servico=ServicoRisco.carregar_arquivo(apoio.ARQUIVO_MODELO),
            repositorio=self.repo,
            config=self.config,
            verificar_token=verificar_token_falso,
        )
        self.cliente = TestClient(self.app)

    def test_status_publico(self):
        resposta = self.cliente.get("/")
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.json()["status"], "online")

    def test_autenticacao_e_perfis(self):
        corpo = {"frequencia": 85, "nota_media": 7.5, "faixa_renda": "media"}
        self.assertEqual(self.cliente.post("/prever-evasao", json=corpo).status_code, 401)
        self.assertEqual(self.cliente.post("/prever-evasao", json=corpo, headers=cabecalho("x")).status_code, 401)
        self.assertEqual(self.cliente.post("/prever-evasao", json=corpo, headers=cabecalho("t-aluno")).status_code, 403)
        resposta = self.cliente.post("/prever-evasao", json=corpo, headers=cabecalho("t-prof"))
        self.assertEqual(resposta.status_code, 200)
        self.assertIn(resposta.json()["classificacao"], {"Baixo", "Médio", "Alto"})
        self.assertEqual(self.cliente.post("/prever-evasao", json=corpo, headers=cabecalho("t-claim")).status_code, 200)

    def test_validacao_de_entrada(self):
        for corpo in (
            {"frequencia": 150, "nota_media": 7},
            {"frequencia": 85, "nota_media": 75},
            {"frequencia": 85, "nota_media": 7, "faixa_renda": "rica"},
        ):
            resposta = self.cliente.post("/prever-evasao", json=corpo, headers=cabecalho("t-prof"))
            self.assertEqual(resposta.status_code, 422, corpo)

    def test_risco_calculado_e_gravado_pela_api(self):
        resposta = self.cliente.post("/alunos/a1/risco", headers=cabecalho("t-prof"))
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.json()["classificacao"], "Alto")  # 45% de frequência, média 3,0
        self.assertEqual(self.repo.alunos["a1"]["classificacaoRisco"], "Alto")
        self.assertEqual(len(self.repo.historico["a1"]), 1)
        self.assertEqual(self.repo.historico["a1"][0]["solicitadoPor"], "u-prof")

    def test_sem_dados_fica_nao_calculado(self):
        resposta = self.cliente.post("/alunos/a2/risco", headers=cabecalho("t-prof")).json()
        self.assertIsNone(resposta["risco"])
        self.assertEqual(self.repo.alunos["a2"]["classificacaoRisco"], "Não calculado")

    def test_aluno_inexistente(self):
        self.assertEqual(self.cliente.post("/alunos/zz/risco", headers=cabecalho("t-prof")).status_code, 404)

    def test_recalculo_em_lote(self):
        self.assertEqual(self.cliente.post("/alunos/risco/recalcular", json={}, headers=cabecalho("t-prof")).status_code, 403)
        resposta = self.cliente.post("/alunos/risco/recalcular", json={"turma": "9º Ano A"}, headers=cabecalho("t-prof"))
        self.assertEqual(resposta.json()["processados"], 2)
        resposta = self.cliente.post("/alunos/risco/recalcular", json={}, headers=cabecalho("t-dir"))
        self.assertEqual(resposta.json()["processados"], 4)

    def test_desfecho_e_aprendido_uma_unica_vez(self):
        self.assertEqual(self.cliente.post("/alunos/a3/desfecho", headers=cabecalho("t-prof")).status_code, 403)
        resposta = self.cliente.post("/alunos/a3/desfecho", headers=cabecalho("t-sec")).json()
        self.assertTrue(resposta["aprendido"])
        self.assertEqual(self.repo.alunos["a3"]["desfechoAprendido"]["rotulo"], 1)
        self.assertIn("atual", self.repo.modelos)  # modelo atualizado vai para o Firestore
        self.assertEqual(resposta["risco"]["classificacao"], "Não calculado")  # evadido sai do painel
        segunda = self.cliente.post("/alunos/a3/desfecho", headers=cabecalho("t-sec")).json()
        self.assertFalse(segunda["aprendido"])

    def test_transferido_nao_e_aprendido_como_evasao(self):
        resposta = self.cliente.post("/alunos/a4/desfecho", headers=cabecalho("t-dir")).json()
        self.assertFalse(resposta["aprendido"])
        self.assertNotIn("atual", self.repo.modelos)

    def test_retreino_publica_modelo_aprovado(self):
        self.assertEqual(self.cliente.post("/modelo/retreinar", headers=cabecalho("t-sec")).status_code, 403)
        self.assertEqual(self.cliente.post("/modelo/retreinar", headers=cabecalho("t-dir")).status_code, 202)
        self.app.state.thread_retreino.join(timeout=120)
        info = self.cliente.get("/modelo", headers=cabecalho("t-dir")).json()
        self.assertEqual(info["retreino"]["estado"], "concluido", info["retreino"])
        self.assertEqual(info["retreino"]["origemDados"]["casos_reais"], 1)  # só o evadido; transferido fica fora
        self.assertIn("atual", self.repo.modelos)

    def test_retreino_reprovado_mantem_o_modelo(self):
        self.config.recall_minimo = 0.999
        versao = self.app.state.servico.versao
        self.cliente.post("/modelo/retreinar", headers=cabecalho("t-dir"))
        self.app.state.thread_retreino.join(timeout=120)
        self.assertEqual(self.app.state.retreino["estado"], "rejeitado")
        self.assertEqual(self.app.state.servico.versao, versao)
        self.assertNotIn("atual", self.repo.modelos)

    def test_cors_so_para_origens_do_sistema(self):
        preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization"}
        for origem, permitido in (
            ("https://sistematcc-4a365.web.app", True),
            ("http://localhost:53211", True),
            ("https://site-qualquer.com", False),
        ):
            resposta = self.cliente.options("/prever-evasao", headers={"Origin": origem, **preflight})
            self.assertEqual(resposta.headers.get("access-control-allow-origin") == origem, permitido, origem)


class TestModoDesenvolvimento(unittest.TestCase):
    def test_sem_autenticacao_e_sem_banco(self):
        app = criar_app(
            servico=ServicoRisco.carregar_arquivo(apoio.ARQUIVO_MODELO),
            config=Configuracao(exigir_autenticacao=False),
            conectar_banco=False,
        )
        cliente = TestClient(app)
        self.assertEqual(cliente.post("/prever-evasao", json={"frequencia": 50, "nota_media": 4}).status_code, 200)
        self.assertEqual(cliente.post("/alunos/a1/risco").status_code, 503)


if __name__ == "__main__":
    unittest.main()
