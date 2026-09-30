# =====================================================================
# 1. TODAS AS IMPORTAÇÕES (Sempre no topo do arquivo)
# =====================================================================
import pickle
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
from apscheduler.schedulers.background import BackgroundScheduler

# =====================================================================
# 2. INICIALIZAÇÃO DO FIREBASE ADMIN
# =====================================================================
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

try:
    firebase_json_env = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    
    if firebase_json_env:
        # Se a variável existe, carrega o JSON diretamente da memória do Render
        cred_dict = json.loads(firebase_json_env)
        cred = credentials.Certificate(cred_dict)
    else:
        # Caso esteja a rodar localmente na sua máquina
        cred = credentials.Certificate("chave_firebase.json")
        
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("☁️ Conectado ao Firebase Firestore com sucesso!")
except Exception as e:
    print(f"❌ Erro ao conectar ao Firebase: {e}")

# ==============================================================================
# 3. CLASSES DA INTELIGÊNCIA ARTIFICIAL (FUZZY ART e ARTMAP)
# ==============================================================================
class FuzzyART:
    def __init__(self, rho=0.85, alpha=0.001, beta=1.0):
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.W = [] 
        self.num_categorias = 0
    
    def _fuzzy_and(self, x, y):
        return np.minimum(x, y)

    def treinar_padrao(self, a):
        I = np.concatenate((a, 1.0 - a))
        if self.num_categorias == 0:
            self.W.append(I)
            self.num_categorias += 1
            return 0 

        T = []
        for j in range(self.num_categorias):
            numerador = np.sum(self._fuzzy_and(I, self.W[j]))
            denominador = self.alpha + np.sum(self.W[j])
            ativacao = numerador / denominador
            T.append(ativacao)

        categorias_ordenadas = np.argsort(T)[::-1]

        for j in categorias_ordenadas:
            match = np.sum(self._fuzzy_and(I, self.W[j])) / np.sum(I)
            if match >= self.rho:
                novo_peso = self.beta * self._fuzzy_and(I, self.W[j]) + (1 - self.beta) * self.W[j]
                self.W[j] = novo_peso
                return j 

        self.W.append(I)
        self.num_categorias += 1
        return self.num_categorias - 1 

class FuzzyARTMAP:
    def __init__(self, rho_a_base=0.5, rho_b=1.0, alpha=0.001, beta=1.0, epsilon=0.001):
        self.rho_a_base = rho_a_base
        self.art_a = FuzzyART(rho=rho_a_base, alpha=alpha, beta=beta)
        self.art_b = FuzzyART(rho=rho_b, alpha=alpha, beta=beta)
        self.map_field = {}
        self.epsilon = epsilon
        self.rotulos_b = {}
    
    def treinar_supervisionado(self, entrada_a, gabarito_b):
        self.art_a.rho = self.rho_a_base
        categoria_k = self.art_b.treinar_padrao(gabarito_b)
        self.rotulos_b[categoria_k] = int(gabarito_b[0])

        match_tracking_ativo = True
        while match_tracking_ativo:
            categoria_j = self.art_a.treinar_padrao(entrada_a)
            if categoria_j not in self.map_field:
                self.map_field[categoria_j] = categoria_k
                match_tracking_ativo = False
            elif self.map_field[categoria_j] == categoria_k:
                match_tracking_ativo = False
            else:
                I = np.concatenate((entrada_a, 1.0 - entrada_a))
                ativacao_atual = np.sum(np.minimum(I, self.art_a.W[categoria_j])) / np.sum(I)
                self.art_a.rho = ativacao_atual + self.epsilon
        return categoria_j, categoria_k

    def prever(self, entrada_a):
        I = np.concatenate((entrada_a, 1.0 - entrada_a))
        T = []
        for j in range(self.art_a.num_categorias):
            numerador = np.sum(np.minimum(I, self.art_a.W[j]))
            denominador = self.art_a.alpha + np.sum(self.art_a.W[j])
            T.append(numerador / denominador)

        if len(T) == 0:
            return 0

        categorias_ordenadas = np.argsort(T)[::-1]

        for j in categorias_ordenadas:
            match = np.sum(np.minimum(I, self.art_a.W[j])) / np.sum(I)
            if match >= self.art_a.rho:
                categoria_k = self.map_field.get(j, -1)
                return self.rotulos_b.get(categoria_k, 0)
        return 0

# =====================================================================
# 4. CONFIGURAÇÃO DO FASTAPI E DO MODELO PREDITIVO
# =====================================================================
class DadosAluno(BaseModel):
    frequencia: float
    nota_media: float
    renda: float

app = FastAPI(
    title="API Fuzzy ARTMAP - Predição de Evasão",
    description="API para o TCC integrando Flutter e Inteligência Artificial"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AdaptadorPickle(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            module = __name__
        return super().find_class(module, name)

try:
    with open('modelo_fuzzy_artmap.pkl', 'rb') as arquivo:
        modelo_artmap = AdaptadorPickle(arquivo).load()
        print("✅ Modelo Fuzzy ARTMAP carregado com sucesso!")
except FileNotFoundError:
    modelo_artmap = None
    print("❌ Erro: Arquivo 'modelo_fuzzy_artmap.pkl' não encontrado.")

# =====================================================================
# 5. ROTAS DA API
# =====================================================================
@app.post("/prever-evasao")
def prever_risco_aluno(aluno: DadosAluno):
    if modelo_artmap is None:
        raise HTTPException(status_code=500, detail="O modelo de IA não foi carregado corretamente.")

    try:
        padrao_entrada = np.array([aluno.frequencia, aluno.nota_media, aluno.renda])
        predicao = modelo_artmap.prever(padrao_entrada)
        resultado_float = float(predicao)
        nivel_risco = "Alto" if resultado_float == 1.0 else "Baixo"

        return {
            "sucesso": True,
            "riscoEvasao": resultado_float,
            "classificacao": nivel_risco,
            "mensagem": f"O aluno apresenta {nivel_risco.lower()} risco de evasão escolar."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar a predição: {str(e)}")

@app.get("/")
def home():
    return {"status": "online", "sistema": "API Fuzzy ARTMAP TCC"}

# =====================================================================
# 6. MÓDULO DE APRENDIZADO CONTÍNUO E AGENDADOR (CRON)
# =====================================================================
def rotina_de_retreinamento():
    global modelo_artmap 
    print(f"\n🔄 [{datetime.now()}] Iniciando re-treinamento automático com dados da nuvem...")
    
    try:
        print("☁️ Baixando base de alunos atualizada do Firebase...")
        colecao_alunos = db.collection('alunos').stream()
        
        X_lista = []
        y_lista = []
        
        for doc in colecao_alunos:
            dados = doc.to_dict()
            
            if dados is not None and 'frequencia' in dados and 'notaMediaGeral' in dados and 'renda' in dados and 'evadiu_historico' in dados:
                freq = float(dados['frequencia']) / 100.0
                nota = float(dados['notaMediaGeral']) / 10.0 
                renda = float(dados['renda']) 
                evasao = int(dados['evadiu_historico']) 
                
                X_lista.append([freq, nota, renda])
                y_lista.append([evasao])
        
        if len(X_lista) < 10:
            print("⚠️ Dados insuficientes no Firebase para re-treino. Rotina abortada.")
            return
            
        X = np.array(X_lista)
        y = np.array(y_lista).reshape(-1, 1) # O reshape previne o erro de concatenação
        
        print(f"📊 {len(X)} alunos baixados. Iniciando o treino da rede...")
        
        novo_modelo = FuzzyARTMAP(rho_a_base=0.65)
        for i in range(len(X)):
            novo_modelo.treinar_supervisionado(X[i], y[i])
            
        with open('modelo_fuzzy_artmap.pkl', 'wb') as arquivo:
            pickle.dump(novo_modelo, arquivo)
            
        modelo_artmap = novo_modelo
        print("✅ Re-treinamento concluído! A IA está mais inteligente com os novos dados do Firebase.\n")
        
    except Exception as e:
        print(f"❌ Falha no re-treinamento automático: {e}\n")

# Ligar o relógio de re-treinamento em segundo plano
agendador = BackgroundScheduler()
agendador.add_job(rotina_de_retreinamento, 'interval', days=30)
agendador.start()