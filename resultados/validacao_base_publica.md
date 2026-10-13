# Validação do método numa base pública com alunos reais

Gerado por `validar_base_publica.py` em 13/10/2026 15:40.

**Base:** *Predict Students' Dropout and Academic Success* (UCI Machine Learning Repository, id 697, licença
CC BY 4.0) — REALINHO, V.; VIEIRA MARTINS, M.; MACHADO, J.; BAPTISTA, L., 2021,
https://doi.org/10.24432/C5MC89. Alunos de **ensino superior em Portugal**; foram mantidos só os
3630 com desfecho conhecido (*Dropout* = evadiu, *Graduate* = concluiu; 39,1% evadiram).
Validação cruzada estratificada, 5 dobras, `random_state=42`.

> Esta base **não** é do mesmo contexto do sistema (educação básica no Brasil) e não tem as mesmas variáveis.
> Ela serve para mostrar que o *método* funciona com dados reais e para compará-lo com outros modelos.

## 1. Três atributos análogos aos do TCC

- **participacao**: fração das disciplinas do 1º semestre com avaliação (1 - sem avaliação / inscritas)
- **desempenho**: média das notas do 1º semestre (escala 0 a 20)
- **socioeconomico**: mensalidades em dia (1) ou atrasadas (0)

| Modelo | Acurácia | Precisão | Recall (evadiu) | F1 | AUC | Observação |
|---|---|---|---|---|---|---|
| Chutar 'ninguém evade' | 60,9% | 0,0% | 0,0% | 0,000 | — |  |
| Regressão logística | 80,7% | 88,9% | 57,8% | 0,701 | 0,852 |  |
| Árvore de decisão (prof. 3) | 82,4% | 90,9% | 61,1% | 0,731 | 0,848 |  |
| Random Forest | 82,7% | 92,5% | 60,7% | 0,733 | 0,872 |  |
| **Comitê de 7 redes Fuzzy ARTMAP** | 76,3% | 69,0% | 71,6% | 0,703 | 0,832 | hiperparâmetros do TCC |

## 2. Todos os atributos numéricos (26)

Inclui desempenho e aprovações dos dois semestres, idade, bolsa, dívidas e indicadores econômicos.
Códigos nominais (curso, nacionalidade, ocupação dos pais) ficaram de fora.

| Modelo | Acurácia | Precisão | Recall (evadiu) | F1 | AUC | Observação |
|---|---|---|---|---|---|---|
| Chutar 'ninguém evade' | 60,9% | 0,0% | 0,0% | 0,000 | — |  |
| Regressão logística | 91,3% | 92,6% | 84,5% | 0,884 | 0,951 |  |
| Árvore de decisão (prof. 3) | 88,9% | 92,9% | 77,6% | 0,845 | 0,910 |  |
| Random Forest | 90,2% | 92,2% | 81,8% | 0,867 | 0,949 |  |
| **Comitê de 7 redes Fuzzy ARTMAP** | 84,5% | 80,4% | 79,9% | 0,802 | 0,901 | hiperparâmetros do TCC |
