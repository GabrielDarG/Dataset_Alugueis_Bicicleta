[README.md](https://github.com/user-attachments/files/29300912/README.md)
# 🚲 Previsão de Aluguéis de Bicicleta (Seoul Bike Sharing)

Modelo de Machine Learning que prevê a **demanda de aluguéis de bicicleta** em Seoul com base em condições climáticas, data e horário. O projeto compara 4 algoritmos de regressão, otimiza o melhor deles com busca de hiperparâmetros e entrega um modelo final pronto para fazer previsões com dados novos.

## 📊 Sobre o projeto

Empresas de bike sharing precisam saber, com antecedência, **quantas bicicletas vão ser alugadas** em determinado dia/hora para conseguir distribuir a frota de forma eficiente. Esse projeto usa dados históricos de Seoul (clima + calendário) para treinar um modelo capaz de estimar essa demanda.

O pipeline cobre o fluxo completo de um projeto de ML:

1. **Carregamento dos dados** direto de um CSV público no GitHub
2. **Limpeza e engenharia de features** (variáveis cíclicas, one-hot encoding, tradução de colunas)
3. **Competição entre 4 modelos**: XGBoost, LightGBM, Random Forest e Gradient Boosting
4. **Otimização de hiperparâmetros** com `RandomizedSearchCV` e `GridSearchCV`
5. **Simulação de uso real**, prevendo a demanda a partir de um novo conjunto de dados

## 🧠 Dataset

Os dados usados são do **Seoul Bike Sharing Demand Dataset**, contendo registros horários de aluguel de bicicletas junto com informações meteorológicas e de calendário.

> Fonte: [SeoulBikeData.csv](https://raw.githubusercontent.com/GabrielDarG/Dataset_Alugueis_Bicicleta/main/SeoulBikeData.csv)

Principais colunas utilizadas (já traduzidas no projeto):

| Coluna | Descrição |
|---|---|
| `Qtd_Aluguel` | Quantidade de bicicletas alugadas (**variável alvo**) |
| `Hora` | Hora do dia (0 a 23) |
| `Temp_C` | Temperatura em °C |
| `Umidade` | Umidade relativa (%) |
| `Vento` | Velocidade do vento (m/s) |
| `Chuva_mm` | Precipitação (mm) |
| `Estacao` | Estação do ano (Primavera, Verão, Outono, Inverno) |
| `Feriado` | Se é feriado ou não |
| `Dia_Funcionamento` | Se o sistema de bicicletas estava operando |
| `Dia_Semana` | Dia da semana |
| `Dia` | Dia do mês |

## ⚙️ Engenharia de features

Alguns destaques do pré-processamento:

- **Features cíclicas (seno/cosseno)** para `Hora` e `Mês`, para que o modelo entenda que 23h está "perto" de 0h e que dezembro está "perto" de janeiro — algo que uma representação numérica simples não capturaria.
- **One-Hot Encoding** para variáveis categóricas (`Estacao`, `Feriado`, `Dia_Funcionamento`, `Dia_Semana`), com `drop_first=True` para evitar multicolinearidade.
- Remoção de colunas pouco relevantes para o problema (ponto de orvalho, visibilidade, radiação solar, neve).

## 🤖 Modelos treinados

O projeto treina e compara, com configurações padrão (*baseline*):

- **XGBoost**
- **LightGBM**
- **Random Forest**
- **Gradient Boosting**

Depois da comparação inicial, o **LightGBM** e o **XGBoost** passam por uma rodada de otimização com `RandomizedSearchCV`, e o melhor deles (LightGBM) recebe um refinamento extra com `GridSearchCV`, usando **RMSLE** (Root Mean Squared Log Error) como métrica principal — ideal para esse tipo de problema, já que penaliza proporcionalmente erros em valores baixos e altos de demanda.

O modelo campeão final é salvo como `modelo_lgbm_refinado.pkl`.

### Métricas utilizadas
- **MSE** (Erro Quadrático Médio)
- **R²** (coeficiente de determinação)
- **RMSLE** (Raiz do Erro Quadrático Logarítmico Médio)

## 📁 Estrutura de arquivos gerados

Ao rodar o script, os seguintes arquivos `.pkl` são criados automaticamente:

| Arquivo | Descrição |
|---|---|
| `colunas_do_modelo.pkl` | Lista das colunas usadas no treino (necessária para alinhar dados novos) |
| `modelo_previsao_bicicletas.pkl` | Modelo XGBoost baseline |
| `modelo_previsao_bicicletas_LGBM.pkl` | Modelo LightGBM baseline |
| `modelo_previsao_bicicletas_RandomForest.pkl` | Modelo Random Forest baseline |
| `modelo_previsao_bicicletas_GradientBoosting.pkl` | Modelo Gradient Boosting baseline |
| `modelo_lgbm_refinado.pkl` | 🏆 Modelo final, otimizado (LightGBM) |

## 🚀 Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
```

### 2. Crie um ambiente virtual (opcional, mas recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências

```bash
pip install pandas numpy joblib xgboost lightgbm scikit-learn
```

### 4. Execute o script

```bash
python Previsão_de_Alugueis_de_Bicicleta.py
```

> ⚠️ O script baixa o dataset automaticamente da internet, então não é necessário ter o CSV localmente. A otimização de hiperparâmetros (`RandomizedSearchCV` e `GridSearchCV`) pode levar alguns minutos para concluir.

## 🔮 Fazendo uma previsão com dados novos

Depois que o modelo é treinado e salvo, você pode prever a demanda para qualquer cenário alterando o dicionário `novos_dados_para_prever`:

```python
novos_dados_para_prever = {
    'Hora': 18,
    'Temp_C': 25.0,
    'Umidade': 60,
    'Vento': 2.2,
    'Chuva_mm': 0.0,
    'Estacao': 'Primavera',
    'Feriado': 'No Holiday',
    'Dia_Funcionamento': 'Yes',
    'Mes': 10,
    'Dia_Semana': 'Quarta',
    'Dia': 15
}
```

A saída será algo como:

```
-------------------------------------------
📈 PREVISÃO DE ALUGUÉIS DE BICICLETA:
Para as condições informadas, a demanda estimada é de aproximadamente 742 bicicletas.
-------------------------------------------
```

A função `preparar_dados()` cuida de toda a transformação necessária (features cíclicas + one-hot encoding + alinhamento de colunas) para que os dados novos fiquem no mesmo formato usado no treinamento.

## 🛠️ Tecnologias utilizadas

- **Python**
- [pandas](https://pandas.pydata.org/) e [NumPy](https://numpy.org/) — manipulação de dados
- [scikit-learn](https://scikit-learn.org/) — Random Forest, Gradient Boosting, métricas e busca de hiperparâmetros
- [XGBoost](https://xgboost.readthedocs.io/) e [LightGBM](https://lightgbm.readthedocs.io/) — modelos de gradient boosting
- [joblib](https://joblib.readthedocs.io/) — serialização dos modelos

## 💡 Possíveis melhorias futuras

- Empacotar o modelo final em uma **API (Flask/FastAPI)** para servir previsões em tempo real
- Criar uma interface web simples para inserir os dados manualmente
- Adicionar validação cruzada mais robusta na etapa de comparação inicial dos modelos
- Versionar os modelos `.pkl` com algo como DVC ao invés de versioná-los direto no Git

## 📄 Licença

Este projeto está disponível para uso livre em fins de estudo e portfólio. Sinta-se à vontade para usar como referência ou ponto de partida para seus próprios projetos de ML! 🚀
