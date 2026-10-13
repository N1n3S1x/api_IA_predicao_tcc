import unittest

import numpy as np

import apoio  # noqa: F401
import preprocessamento as pp


class TestNormalizacao(unittest.TestCase):
    def test_entrada_na_escala_do_modelo(self):
        np.testing.assert_allclose(pp.normalizar_entrada(85, 7.5, "media"), [0.85, 0.75, 0.5])

    def test_valores_fora_da_faixa_sao_rejeitados(self):
        for frequencia, nota in [(120, 7), (-1, 7), (80, 11), (80, -0.5)]:
            with self.assertRaises(pp.DadosInvalidos):
                pp.normalizar_entrada(frequencia, nota, "baixa")

    def test_faixa_de_renda_em_varios_formatos(self):
        casos = {
            "Média": "media",
            "BAIXA": "baixa",
            "alta": "alta",
            1500.0: "baixa",  # registro antigo em reais (< 2 salários mínimos)
            5000.0: "media",
            9000.0: "alta",
            0.0: "baixa",  # já codificado na escala do modelo
            0.5: "media",
            1.0: "alta",
            None: None,
            "": None,
        }
        for valor, esperado in casos.items():
            self.assertEqual(pp.normalizar_faixa_renda(valor), esperado, valor)

    def test_faixa_de_renda_invalida(self):
        for valor in ["muito alta", -10, True, [1]]:
            with self.assertRaises(pp.DadosInvalidos):
                pp.normalizar_faixa_renda(valor)


class TestRotulos(unittest.TestCase):
    def test_so_desfechos_conhecidos_viram_rotulo(self):
        self.assertEqual(pp.rotulo_do_status("Evadido"), 1)
        self.assertEqual(pp.rotulo_do_status("Concluído"), 0)
        self.assertEqual(pp.rotulo_do_status("concluido"), 0)
        self.assertIsNone(pp.rotulo_do_status("Transferido"))  # transferência não é evasão
        self.assertIsNone(pp.rotulo_do_status("Ativo"))  # desfecho ainda desconhecido
        self.assertIsNone(pp.rotulo_do_status(None))

    def test_aluno_sem_status_conta_como_ativo(self):
        self.assertTrue(pp.esta_ativo(None))
        self.assertTrue(pp.esta_ativo("Ativo"))
        self.assertFalse(pp.esta_ativo("Evadido"))


class TestExtracao(unittest.TestCase):
    def test_frequencia_vem_dos_contadores_de_chamada(self):
        ind = pp.extrair_indicadores({"aulasRegistradas": 40, "faltas": 10, "frequencia": 99, "notaMediaGeral": 6})
        self.assertEqual(ind.frequencia, 75.0)

    def test_registro_antigo_com_frequencia_e_renda_em_reais(self):
        ind = pp.extrair_indicadores({"frequencia": 92.5, "notaMediaGeral": 8.25, "renda": 1500.0})
        self.assertEqual((ind.frequencia, ind.nota_media, ind.faixa_renda), (92.5, 8.25, "baixa"))
        self.assertTrue(ind.completos)

    def test_faixa_de_renda_tem_prioridade_sobre_a_renda_antiga(self):
        ind = pp.extrair_indicadores({"frequencia": 90, "notaMediaGeral": 8, "faixaRenda": "alta", "renda": 900})
        self.assertEqual(ind.faixa_renda, "alta")

    def test_aponta_o_que_falta(self):
        ind = pp.extrair_indicadores({"faixaRenda": "baixa"})
        self.assertEqual(ind.faltando, ["frequência", "nota média"])
        self.assertFalse(ind.completos)

    def test_valor_absurdo_no_banco_e_rejeitado(self):
        with self.assertRaises(pp.DadosInvalidos):
            pp.extrair_indicadores({"frequencia": 150, "notaMediaGeral": 7})


class TestPerfis(unittest.TestCase):
    def test_variantes_gravadas_no_banco(self):
        for valor, esperado in {"direcao": "direcao", "Diretor": "direcao", "Direção": "direcao",
                                "Secretário": "secretaria", "professora": "professor", "aluno": "aluno",
                                "Funcionário": None, None: None}.items():
            self.assertEqual(pp.normalizar_perfil(valor), esperado, valor)


class TestFatores(unittest.TestCase):
    def test_fatores_em_linguagem_simples(self):
        fatores = " ".join(pp.fatores_de_atencao(60, 4.0, "baixa"))
        self.assertIn("mínimo legal de 75%", fatores)
        self.assertIn("recuperação", fatores)
        self.assertIn("renda familiar", fatores)

    def test_renda_nao_informada_e_avisada(self):
        self.assertIn("não informada", " ".join(pp.fatores_de_atencao(95, 9, None)))

    def test_aluno_sem_alertas(self):
        self.assertEqual(pp.fatores_de_atencao(95, 9, "alta"), [])


if __name__ == "__main__":
    unittest.main()
