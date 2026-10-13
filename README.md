# Modelo e API de IA

| Arquivo | O que faz |
|---|---|
| `fuzzy_artmap.py` | Fuzzy ARTMAP (rede única) e comitê de redes, com serialização em JSON |
| `preprocessamento.py` | Normalização e regras de negócio (faixas de renda, rótulos de desfecho, perfis) |
| `servico_risco.py` | Calcula o risco de um aluno, explica o resultado e aprende novos casos |
| `aplicacao.py` / `api.py` | API FastAPI (autenticação, perfis, rotas) e ponto de entrada do Render |
| `repositorio.py` | Acesso ao Firestore pelo Admin SDK |
| `gerar_dados_ficticios.py` | Gera a base sintética `dados/dados_alunos_tcc.csv` e calcula o teto teórico |
| `treinar_modelo.py` | Avalia os modelos, gera `resultados/` e o modelo publicado `modelo/modelo_fuzzy_artmap.json` |
| `validar_base_publica.py` | Avalia o método numa base pública com alunos reais (UCI) |
| `criar_usuarios_teste.py` | Contas de teste, alunos de demonstração e sincronização de perfis (custom claims) |

## Comandos (dentro desta pasta)

```bash
pip install -r requirements-dev.txt     # requirements.txt tem só o necessário para a API
python treinar_modelo.py                # avaliação + figuras + modelo de produção
python validar_base_publica.py          # baixa a base da UCI na 1ª execução
python -m unittest discover -s tests    # testes
uvicorn api:app --reload                # API local; documentação em http://127.0.0.1:8000/docs
```

A API procura a chave da conta de serviço em `FIREBASE_CREDENTIALS_JSON` ou em `chave_firebase.json` (esta pasta,
fora do git). Para testar sem login: `EXIGIR_AUTENTICACAO=false uvicorn api:app --reload`.

## Rotas

| Rota | Quem pode | O que faz |
|---|---|---|
| `GET /` | todos | status (o app usa para "acordar" o Render) |
| `POST /prever-evasao` | equipe | simulação com valores informados (não grava) |
| `POST /alunos/{id}/risco` | equipe | calcula e grava o risco de um aluno |
| `POST /alunos/risco/recalcular` | equipe (gestão para todos) | recalcula vários alunos |
| `POST /alunos/{id}/desfecho` | gestão | situação mudou: aprende o desfecho e atualiza o risco |
| `GET /modelo` | gestão | versão, métricas e estado do re-treino |
| `POST /modelo/retreinar` | direção | re-treino completo; só publica se recall e F1 ≥ 70% |

O modelo publicado fica no Firestore (`modelos_ia/atual`, com backup em `modelos_ia/anterior`), porque o disco do
Render é apagado a cada reinício.
