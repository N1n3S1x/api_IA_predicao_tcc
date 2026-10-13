import json
import unittest

import apoio
from servico_risco import (
    NAO_CALCULADO,
    ServicoRisco,
    carregar_base_csv,
    classificar,
    treinar_servico,
    validacao_cruzada,
)

ATIVO = {"statusMatricula": "Ativo"}


class TestServicoComModeloPublicado(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servico = ServicoRisco.carregar_arquivo(apoio.ARQUIVO_MODELO)

    def test_perfis_extremos(self):
        self.assertEqual(self.servico.avaliar(30, 2.0, "baixa").classificacao, "Alto")
        self.assertEqual(self.servico.avaliar(98, 9.5, "alta").classificacao, "Baixo")

    def test_regressao_do_bug_da_renda(self):
        """Na versão anterior este aluno (40%, nota 2,0, R$ 1.200) saía "Baixo"
        porque o app enviava a renda dividida por 5000 e o modelo foi treinado
        com faixas 0 / 0,5 / 1."""
        doc = {**ATIVO, "frequencia": 40, "notaMediaGeral": 2.0, "renda": 1200.0}
        resultado = self.servico.avaliar_documento(doc)
        self.assertEqual(resultado.classificacao, "Alto")
        self.assertEqual(resultado.entrada["faixaRenda"], "baixa")

    def test_resultado_tem_score_explicacao_e_versao(self):
        resultado = self.servico.avaliar(70, 5.5, "media")
        self.assertTrue(0.0 <= resultado.risco <= 1.0)
        self.assertEqual(resultado.classificacao, classificar(resultado.risco))
        self.assertIn("perfil aprendido", resultado.perfil)
        self.assertEqual(resultado.versao_modelo, self.servico.versao)

    def test_falta_de_dados_nunca_vira_baixo_risco(self):
        resultado = self.servico.avaliar_documento({**ATIVO, "faixaRenda": "baixa"})
        self.assertIsNone(resultado.risco)
        self.assertEqual(resultado.classificacao, NAO_CALCULADO)
        self.assertIn("frequência", resultado.motivo)

    def test_aluno_inativo_nao_recebe_risco(self):
        doc = {"statusMatricula": "Evadido", "frequencia": 40, "notaMediaGeral": 2.0, "faixaRenda": "baixa"}
        self.assertEqual(self.servico.avaliar_documento(doc).classificacao, NAO_CALCULADO)

    def test_renda_nao_informada_ainda_permite_o_calculo(self):
        resultado = self.servico.avaliar_documento({**ATIVO, "frequencia": 50, "notaMediaGeral": 3.0})
        self.assertIsNotNone(resultado.risco)
        self.assertTrue(any("não informada" in f for f in resultado.fatores))

    def test_campos_gravados_no_firestore(self):
        campos = self.servico.avaliar(80, 6.0, "media").campos_aluno()
        self.assertEqual(
            set(campos),
            {"riscoEvasao", "classificacaoRisco", "fatoresRisco", "perfilRisco", "motivoRisco", "versaoModelo"},
        )

    def test_modelo_salvo_e_aceito_pelo_firestore(self):
        dados = self.servico.para_dict()
        self.assertFalse(apoio.tem_lista_aninhada(dados))
        json.dumps(dados)


class TestCasosDeTreino(unittest.TestCase):
    def test_conversao_de_desfechos(self):
        base = {"frequencia": 50, "notaMediaGeral": 4.0, "faixaRenda": "baixa"}
        (x, rotulo), _ = ServicoRisco.caso_de_treino({**base, "statusMatricula": "Evadido"})
        self.assertEqual(rotulo, 1)
        self.assertEqual(x.tolist(), [0.5, 0.4, 0.0])
        (_, rotulo), _ = ServicoRisco.caso_de_treino({**base, "statusMatricula": "Concluído"})
        self.assertEqual(rotulo, 0)
        for status in ("Transferido", "Ativo"):
            caso, motivo = ServicoRisco.caso_de_treino({**base, "statusMatricula": status})
            self.assertIsNone(caso)
            self.assertIn("Desfecho desconhecido", motivo)

    def test_caso_sem_renda_nao_entra_no_treino(self):
        caso, motivo = ServicoRisco.caso_de_treino({"statusMatricula": "Evadido", "frequencia": 50, "notaMediaGeral": 4})
        self.assertIsNone(caso)
        self.assertIn("renda", motivo)

    def test_aprendizado_incremental_atualiza_metadados(self):
        servico = ServicoRisco.carregar_arquivo(apoio.ARQUIVO_MODELO)
        versao = servico.versao
        caso, _ = ServicoRisco.caso_de_treino(
            {"statusMatricula": "Evadido", "frequencia": 99, "notaMediaGeral": 9.9, "faixaRenda": "alta"}
        )
        servico.aprender_caso(*caso)
        self.assertEqual(servico.metadados["casos_incrementais"], 1)
        self.assertNotEqual(servico.versao, versao)
        self.assertEqual(servico.avaliar(99, 9.9, "alta").classificacao, "Alto")


class TestTreino(unittest.TestCase):
    def test_validacao_cruzada_e_treino_na_base(self):
        X, y = carregar_base_csv(apoio.PASTA_IA / "dados" / "dados_alunos_tcc.csv")
        metricas = validacao_cruzada(X, y)
        self.assertGreater(metricas["recall"], 0.75)
        self.assertGreater(metricas["f1"], 0.75)
        servico = treinar_servico(X[:300], y[:300], {"base_sintetica": 300}, metricas, n_redes=3)
        copia = ServicoRisco.de_dict(json.loads(json.dumps(servico.para_dict())))
        self.assertEqual(copia.avaliar(60, 5, "media").risco, servico.avaliar(60, 5, "media").risco)


if __name__ == "__main__":
    unittest.main()
