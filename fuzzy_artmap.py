"""Rede neural Fuzzy ARTMAP (Carpenter et al., 1992) para predição de evasão.

Referência: CARPENTER, G. A.; GROSSBERG, S.; MARKUZON, N.; REYNOLDS, J. H.;
ROSEN, D. B. Fuzzy ARTMAP: A neural network architecture for incremental
supervised learning of analog multidimensional maps. IEEE Transactions on
Neural Networks, v. 3, n. 5, p. 698-713, 1992.

Diferenças em relação à primeira versão do TCC (as duas opções ``*_versao_anterior``
reproduzem o comportamento antigo apenas para a comparação "antes x depois"):

1. Os pesos da categoria vencedora do ARTa só são atualizados DEPOIS que o map
   field confirma o rótulo (ressonância). Antes, a categoria aprendia o padrão
   antes da checagem e, quando o rótulo estava errado, já tinha absorvido um
   aluno da outra classe.
2. A predição usa a vigilância base. Antes, ela usava a vigilância que "sobrava"
   do match-tracking do último aluno do treino e, quando nenhuma categoria
   passava, respondia "não evade" por padrão.
"""

from __future__ import annotations

import numpy as np


class FuzzyARTMAP:
    """Fuzzy ARTMAP com map field simplificado (cada categoria do ARTa guarda o
    seu rótulo, equivalente a um ARTb com vigilância 1,0 para rótulos discretos).

    :param n_entradas: número de atributos de entrada (já normalizados em [0, 1]).
    :param rho_a: vigilância base do ARTa (0 a 1). Controla a similaridade exigida.
    :param alpha: parâmetro de escolha (> 0). Evita divisão por zero.
    :param beta: taxa de aprendizagem (0 a 1). 1,0 é o aprendizado rápido.
    :param epsilon: incremento da vigilância no match-tracking.
    :param aprender_antes_do_map_field: reproduz o defeito 1 da versão anterior
        (aprende antes de checar o rótulo). Use só para comparação.
    :param predicao_versao_anterior: reproduz o defeito 2 da versão anterior
        (vigilância residual do treino e "não evade" por padrão). Só para comparação.
    """

    def __init__(
        self,
        n_entradas: int = 3,
        rho_a: float = 0.65,
        alpha: float = 0.001,
        beta: float = 1.0,
        epsilon: float = 0.001,
        aprender_antes_do_map_field: bool = False,
        predicao_versao_anterior: bool = False,
    ):
        self.n_entradas = n_entradas
        self.rho_a = rho_a
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.aprender_antes_do_map_field = aprender_antes_do_map_field
        self.predicao_versao_anterior = predicao_versao_anterior
        self._rho_residual = rho_a  # vigilância ao fim do último treino (só p/ comparação)

        # Pesos das categorias (nós F2) em código complemento: [u, 1 - v],
        # onde [u, v] é o hiper-retângulo que a categoria representa.
        self.W = np.empty((0, 2 * n_entradas))
        self.rotulos = np.empty(0, dtype=int)  # map field: rótulo de cada categoria
        self.contagem = np.empty(0, dtype=int)  # alunos de treino que ressoaram nela

    @property
    def n_categorias(self) -> int:
        return len(self.rotulos)

    # ------------------------------------------------------------------
    # Operações básicas
    # ------------------------------------------------------------------
    def _codificar(self, a) -> np.ndarray:
        """Camada F0: código complemento I = [a, 1 - a]."""
        a = np.asarray(a, dtype=float)
        if a.shape != (self.n_entradas,):
            raise ValueError(f"Esperava {self.n_entradas} atributos, recebi {a.shape}.")
        if np.any(a < 0.0) or np.any(a > 1.0):
            raise ValueError("A entrada deve estar normalizada entre 0 e 1.")
        return np.concatenate((a, 1.0 - a))

    def _ativacoes(self, I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Função de escolha T_j e grau de match |I ∧ W_j| / |I| de cada categoria."""
        intersecao = np.minimum(I, self.W).sum(axis=1)  # norma L1 do fuzzy AND
        T = intersecao / (self.alpha + self.W.sum(axis=1))
        match = intersecao / I.sum()
        return T, match

    def _aprender(self, j: int, I: np.ndarray) -> None:
        self.W[j] = self.beta * np.minimum(I, self.W[j]) + (1 - self.beta) * self.W[j]

    def _nova_categoria(self, I: np.ndarray, rotulo: int) -> int:
        self.W = np.vstack([self.W, I])
        self.rotulos = np.append(self.rotulos, rotulo)
        self.contagem = np.append(self.contagem, 1)
        return self.n_categorias - 1

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------
    def treinar_amostra(self, a, rotulo) -> int:
        """Treina com um aluno (aprendizado incremental). Retorna a categoria usada."""
        I = self._codificar(a)
        rotulo = int(rotulo)
        rho = self.rho_a  # a vigilância volta ao valor base a cada novo aluno

        while self.n_categorias:
            T, match = self._ativacoes(I)
            # Busca: categorias em ordem decrescente de ativação que passam na vigilância
            j = next((int(c) for c in np.argsort(T)[::-1] if match[c] >= rho), None)
            if j is None:
                break

            if self.aprender_antes_do_map_field:
                self._aprender(j, I)  # comportamento da versão anterior (só para comparação)

            if self.rotulos[j] == rotulo:
                # Ressonância no map field: agora sim a categoria aprende
                if not self.aprender_antes_do_map_field:
                    self._aprender(j, I)
                self.contagem[j] += 1
                self._rho_residual = rho
                return j

            # Match-tracking: eleva a vigilância logo acima do match da categoria
            # que errou o rótulo e continua a busca. A categoria NÃO é alterada.
            match_j = np.minimum(I, self.W[j]).sum() / I.sum()
            rho = match_j + self.epsilon

        # Nenhuma categoria serviu: cria uma nova associada a este rótulo
        self._rho_residual = rho
        return self._nova_categoria(I, rotulo)

    def treinar(self, X, y, ordem=None) -> "FuzzyARTMAP":
        """Treina com vários alunos na ordem dada (padrão: a ordem dos dados)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        for i in range(len(X)) if ordem is None else ordem:
            self.treinar_amostra(X[i], y[i])
        return self

    # ------------------------------------------------------------------
    # Predição e explicação
    # ------------------------------------------------------------------
    def categoria_vencedora(self, a) -> tuple[int | None, bool]:
        """Categoria escolhida para o aluno e se ela passou na vigilância.

        Se nenhuma categoria passar, usa a de maior ativação (a mais próxima)
        em vez de assumir "não evade" por padrão.
        """
        I = self._codificar(a)
        if not self.n_categorias:
            return None, False
        T, match = self._ativacoes(I)
        ordem = np.argsort(T)[::-1]
        rho = self._rho_residual if self.predicao_versao_anterior else self.rho_a
        j = next((int(c) for c in ordem if match[c] >= rho), None)
        if j is not None:
            return j, True
        if self.predicao_versao_anterior:
            return None, False  # a versão anterior respondia "não evade" aqui
        return int(ordem[0]), False

    def prever(self, a) -> int:
        j, _ = self.categoria_vencedora(a)
        return 0 if j is None else int(self.rotulos[j])

    def prever_lote(self, X) -> np.ndarray:
        return np.array([self.prever(a) for a in np.asarray(X, dtype=float)], dtype=int)

    def caixa(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Hiper-retângulo [inferior, superior] da categoria j, na escala [0, 1]."""
        inferior = self.W[j, : self.n_entradas]
        superior = 1.0 - self.W[j, self.n_entradas :]
        return inferior.copy(), superior.copy()

    # ------------------------------------------------------------------
    # Serialização (JSON / Firestore: listas planas, sem arrays aninhados)
    # ------------------------------------------------------------------
    def para_dict(self) -> dict:
        return {
            "W": [float(v) for v in self.W.ravel()],
            "rotulos": [int(v) for v in self.rotulos],
            "contagem": [int(v) for v in self.contagem],
        }

    def carregar_pesos(self, dados: dict) -> "FuzzyARTMAP":
        dim = 2 * self.n_entradas
        self.W = np.asarray(dados["W"], dtype=float).reshape(-1, dim)
        self.rotulos = np.asarray(dados["rotulos"], dtype=int)
        self.contagem = np.asarray(dados["contagem"], dtype=int)
        if not (len(self.W) == len(self.rotulos) == len(self.contagem)):
            raise ValueError("Pesos, rótulos e contagens com tamanhos diferentes.")
        return self


class ComiteFuzzyARTMAP:
    """Votação de várias redes Fuzzy ARTMAP treinadas em ordens diferentes.

    O resultado do ARTMAP depende da ordem de apresentação dos alunos. A votação
    (estratégia sugerida por Carpenter et al., 1992) reduz essa variação e gera
    um score contínuo: a fração de redes que prevê evasão.
    """

    FORMATO = "comite-fuzzy-artmap/1"

    def __init__(self, n_redes: int = 7, semente: int = 42, **parametros_rede):
        if n_redes < 1:
            raise ValueError("O comitê precisa de pelo menos uma rede.")
        self.n_redes = n_redes
        self.semente = semente
        self.parametros_rede = {
            "n_entradas": 3,
            "rho_a": 0.65,
            "alpha": 0.001,
            "beta": 1.0,
            "epsilon": 0.001,
            **parametros_rede,
        }
        self.redes: list[FuzzyARTMAP] = []

    def _nova_rede(self) -> FuzzyARTMAP:
        return FuzzyARTMAP(**self.parametros_rede)

    def treinar(self, X, y) -> "ComiteFuzzyARTMAP":
        """A primeira rede usa a ordem original dos dados; as demais, permutações."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.semente)
        self.redes = []
        for r in range(self.n_redes):
            ordem = np.arange(len(X)) if r == 0 else rng.permutation(len(X))
            self.redes.append(self._nova_rede().treinar(X, y, ordem))
        return self

    def aprender(self, a, rotulo) -> None:
        """Aprendizado incremental de um novo caso confirmado (sem re-treinar tudo)."""
        for rede in self.redes:
            rede.treinar_amostra(a, rotulo)

    def votos(self, a) -> np.ndarray:
        return np.array([rede.prever(a) for rede in self.redes], dtype=int)

    def score(self, a) -> float:
        """Fração das redes que preveem evasão (0 a 1)."""
        return float(self.votos(a).mean())

    def score_lote(self, X) -> np.ndarray:
        return np.array([self.score(a) for a in np.asarray(X, dtype=float)])

    def prever(self, a) -> int:
        return int(self.score(a) >= 0.5)

    def prever_lote(self, X) -> np.ndarray:
        return (self.score_lote(X) >= 0.5).astype(int)

    def explicar(self, a) -> dict:
        """Perfil aprendido que justifica a decisão da maioria.

        Entre as redes que votaram com a maioria, escolhe a categoria vencedora
        com mais alunos de treino e devolve o hiper-retângulo dela.
        """
        votos = self.votos(a)
        rotulo = int(votos.mean() >= 0.5)
        melhor = None
        for rede, voto in zip(self.redes, votos):
            if voto != rotulo:
                continue
            j, passou = rede.categoria_vencedora(a)
            if j is None:
                continue
            if melhor is None or rede.contagem[j] > melhor[0].contagem[melhor[1]]:
                melhor = (rede, j, passou)
        if melhor is None:
            return {"rotulo": rotulo, "inferior": None, "superior": None, "alunos": 0, "dentro_da_vigilancia": False}
        rede, j, passou = melhor
        inferior, superior = rede.caixa(j)
        return {
            "rotulo": rotulo,
            "inferior": inferior.tolist(),
            "superior": superior.tolist(),
            "alunos": int(rede.contagem[j]),
            "dentro_da_vigilancia": bool(passou),
        }

    @property
    def n_categorias(self) -> list[int]:
        return [rede.n_categorias for rede in self.redes]

    def para_dict(self) -> dict:
        return {
            "formato": self.FORMATO,
            "n_redes": self.n_redes,
            "semente": self.semente,
            "parametros_rede": dict(self.parametros_rede),
            "redes": [rede.para_dict() for rede in self.redes],
        }

    @classmethod
    def de_dict(cls, dados: dict) -> "ComiteFuzzyARTMAP":
        if dados.get("formato") != cls.FORMATO:
            raise ValueError(f"Formato de modelo desconhecido: {dados.get('formato')!r}")
        comite = cls(n_redes=dados["n_redes"], semente=dados["semente"], **dados["parametros_rede"])
        comite.redes = [comite._nova_rede().carregar_pesos(r) for r in dados["redes"]]
        if len(comite.redes) != comite.n_redes:
            raise ValueError("Número de redes não confere com o modelo salvo.")
        return comite
