# FIAP Tech Challenge 3 — Atrasos de Voos

Pipeline de ciência de dados em Python para analisar e prever atrasos de voos (>15 minutos) usando o dataset público de voos dos EUA. O fluxo foi reorganizado em notebooks menores, todos em português, com amostra processada para execução rápida.

## Estrutura
- `data/`: CSVs originais (`flights.csv`, `airlines.csv`, `airports.csv`).
- `data/processed/flights_sample.parquet`: amostra limpa (300k voos, sem cancelados/desviados) com features básicas.
- `notebooks/`:
  - `01_preparacao_dados.ipynb`: carrega CSVs, limpa cancelados/divertidos, cria features e salva amostra.
  - `02_eda.ipynb`: exploração descritiva (ausências, distribuição de atrasos, padrões por cia/dia/hora).
  - `03_modelagem_supervisionada.ipynb`: regressão logística (baseline) e Random Forest com pré-processamento robusto.
  - `04_clusterizacao.ipynb`: K-Means para perfis operacionais sem usar o alvo.
  - `05_export_modelo.ipynb`: re-treino do Random Forest e salvamento em `models/random_forest_delay_model.pkl`.
- `models/`: modelos treinados (gerado pelo notebook 05).
- `reports/`: PDFs do enunciado e dicionário de dados.

## Requisitos
Instale dependências no ambiente virtual:
```bash
python -m pip install -r requirements.txt
```

## Como rodar
1. Execute `notebooks/01_preparacao_dados.ipynb` para gerar/atualizar a amostra em `data/processed/`.
2. Rode `02_eda.ipynb` e `03_modelagem_supervisionada.ipynb` para análises e métricas.
3. (Opcional) `04_clusterizacao.ipynb` para agrupamentos.
4. `05_export_modelo.ipynb` para salvar o modelo final em `models/random_forest_delay_model.pkl`.

## Notas
- Variável-alvo: `DELAYED = 1` se `ARRIVAL_DELAY > 15`.
- Features só usam dados conhecidos antes da decolagem para evitar vazamento.
- Pré-processamento inclui imputação e one-hot encoding; métricas principais: precisão, recall, F1 e ROC-AUC.


## Resultados atuais (holdout 20%)
- Random Forest balanceado (tuning leve): ROC-AUC ~0.663.
- Threshold escolhido para F1 da classe de atraso: ~0.435.
- Com esse cutoff: precisao ~0.26 / recall ~0.60 (acc ~0.63), refletindo o desbalanceamento de ~18% atrasos.
- O pickle em `models/random_forest_delay_model.pkl` salva `model` e `threshold` para reproduzir o corte.

## Limitacoes e proximos passos
- Classe de atraso desbalanceada (~18%); com `class_weight="balanced"` e threshold ajustado o recall melhora, mas o ROC-AUC (~0.65) indica espaco para modelos mais fortes.
- Faltam variaveis de contexto (clima, sazonalidade fina, conexoes) que poderiam aumentar o sinal; vale explorar fontes externas e novas features.
- Avaliar calibracao de probabilidade e hiperparametros (ex.: search mais amplo, XGBoost) e comparar abordagens de balanceamento (class_weight vs. reamostragem) priorizando recall sem inflar falsos positivos.
- Para producao, versionar o threshold salvo no modelo e monitorar drift de distribuicoes e degradacao de recall.
