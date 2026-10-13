import json
import unittest

import numpy as np

import apoio  # noqa: F401  (ajusta o sys.path)
from fuzzy_artmap import ComiteFuzzyARTMAP, FuzzyARTMAP


class TestFuzzyARTMAP(unittest.TestCase):
    def test_match_tracking_nao_altera_a_categoria_que_errou(self):
        rede = FuzzyARTMAP(rho_a=0.5)
        rede.treinar_amostra([0.2, 0.2, 0.2], 0)
        pesos_antes = rede.W[0].copy()
        rede.treinar_amostra([0.3, 0.3, 0.3], 1)  # parecido, mas com o rótulo oposto
        np.testing.assert_array_equal(rede.W[0], pesos_antes)
        self.assertEqual(rede.n_categorias, 2)

    def test_versao_anterior_alterava_a_categoria_antes_do_map_field(self):
        rede = FuzzyARTMAP(rho_a=0.5, aprender_antes_do_map_field=True)
        rede.treinar_amostra([0.2, 0.2, 0.2], 0)
        pesos_antes = rede.W[0].copy()
        rede.treinar_amostra([0.3, 0.3, 0.3], 1)
        self.assertFalse(np.array_equal(rede.W[0], pesos_antes))

    def test_sem_categoria_na_vigilancia_usa_a_mais_proxima(self):
        rede = FuzzyARTMAP(rho_a=0.99).treinar([[0.9, 0.9, 0.9]], [1])
        self.assertEqual(rede.prever([0.1, 0.1, 0.1]), 1)
        antiga = FuzzyARTMAP(rho_a=0.99, predicao_versao_anterior=True).treinar([[0.9, 0.9, 0.9]], [1])
        self.assertEqual(antiga.prever([0.1, 0.1, 0.1]), 0)  # o "não evade" por padrão

    def test_reconhece_os_padroes_de_treino_separaveis(self):
        X = [[0.9, 0.9, 1.0], [0.85, 0.8, 0.5], [0.3, 0.2, 0.0], [0.35, 0.3, 0.5]]
        y = [0, 0, 1, 1]
        rede = FuzzyARTMAP().treinar(X, y)
        self.assertEqual(rede.prever_lote(X).tolist(), y)

    def test_caixa_da_categoria_contem_os_alunos_que_ela_aprendeu(self):
        rng = np.random.default_rng(1)
        X = rng.random((60, 3))
        y = (X[:, 0] < 0.5).astype(int)
        rede = FuzzyARTMAP()
        categorias = [rede.treinar_amostra(x, r) for x, r in zip(X, y)]
        for x, j in zip(X, categorias):
            inferior, superior = rede.caixa(j)
            self.assertTrue(np.all(inferior - 1e-12 <= x) and np.all(x <= superior + 1e-12))

    def test_entrada_fora_de_zero_a_um_e_rejeitada(self):
        with self.assertRaises(ValueError):
            FuzzyARTMAP().treinar_amostra([85, 0.7, 0.5], 0)


class TestComite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.X = rng.random((200, 3))
        cls.y = ((cls.X[:, 0] + cls.X[:, 1]) < 0.9).astype(int)
        cls.comite = ComiteFuzzyARTMAP(n_redes=5, semente=3).treinar(cls.X, cls.y)

    def test_score_e_deterministico_e_entre_zero_e_um(self):
        outro = ComiteFuzzyARTMAP(n_redes=5, semente=3).treinar(self.X, self.y)
        scores = self.comite.score_lote(self.X[:30])
        np.testing.assert_array_equal(scores, outro.score_lote(self.X[:30]))
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_serializacao_ida_e_volta_preserva_as_predicoes(self):
        dados = json.loads(json.dumps(self.comite.para_dict()))
        copia = ComiteFuzzyARTMAP.de_dict(dados)
        pontos = np.random.default_rng(0).random((50, 3))
        np.testing.assert_array_equal(copia.score_lote(pontos), self.comite.score_lote(pontos))

    def test_formato_desconhecido_e_rejeitado(self):
        with self.assertRaises(ValueError):
            ComiteFuzzyARTMAP.de_dict({"formato": "pickle"})

    def test_aprendizado_incremental_de_um_caso_novo(self):
        comite = ComiteFuzzyARTMAP.de_dict(self.comite.para_dict())
        ponto = np.array([0.95, 0.95, 0.5])  # região onde todos preveem "não evade"
        self.assertEqual(comite.prever(ponto), 0)
        comite.aprender(ponto, 1)
        self.assertEqual(comite.prever(ponto), 1)

    def test_explicacao_traz_o_perfil_da_maioria(self):
        explicacao = self.comite.explicar(self.X[0])
        self.assertEqual(explicacao["rotulo"], self.comite.prever(self.X[0]))
        self.assertGreaterEqual(explicacao["alunos"], 1)
        self.assertTrue(all(i <= s for i, s in zip(explicacao["inferior"], explicacao["superior"])))


if __name__ == "__main__":
    unittest.main()
