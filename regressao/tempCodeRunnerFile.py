import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess(df):
    # Selecionar colunas relevantes para regressão
    df = df[['Day', 'Age', 'GallusWeight', 'GallusEggWeight', 'AmountOfFeed','EggsPerDay' , 'SunLightExposure', 'GallusPlumage', 'GallusCombType', 'GallusEggColor', 'GallusBreed']]

    # Definir recursos (features) e alvo (target)
    X = df.drop('GallusWeight', axis=1)
    y = df['GallusWeight']

    # Identificar colunas numéricas
    numeric_features = ['Day', 'Age', 'GallusEggWeight', 'AmountOfFeed', 'SunLightExposure']
    categorical_features = ['GallusPlumage', 'GallusCombType', 'GallusEggColor', 'GallusBreed']

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # 
    ])

# Criar o pré-processador usando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler() , numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor

def model_evaluation(model, X_train, y_train, X_test, y_test):
    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, mae, r2, y_pred

def plot_results(y_test, y_pred, name, config):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('Regressão Linear: Valores Reais vs. Preditos - ' + name + ' - Configuração ' + str(config))
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)  # Linha de referência
    plt.grid(True)

    # Salvar a figura
    if name == 'Regressão Linear':
        plt.savefig(f'./regressao/saidas/regressao/{name}_config_{config}.png')
    elif name == 'KNN Regressor':
        plt.savefig(f'./regressao/saidas/knn/{name}_config_{config}.png')
    elif name == 'Árvore de Decisão Regressora':
        plt.savefig(f'./regressao/saidas/arvore/{name}_config_{config}.png')
    else:
        plt.savefig(f'./regressao/saidas/mlp/{name}_config_{config}.png')
    plt.show()
    
def print_results(name, mse, rmse, mae, r2):
    print(f"{name}:")
    print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
    print(f"Erro Absoluto Médio (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    print()
    save_results(name, mse, rmse, mae, r2)

def save_results(name, mse, rmse, mae, r2):
    results_str = (
        f"{name}: \n"
        f"Erro Quadratico Medio (MSE): {mse:.2f}\n"
        f"Raiz do Erro Quadratico Medio (RMSE): {rmse:.2f}\n"
        f"Erro Absoluto Medio (MAE): {mae:.2f}\n"
        f"R² Score: {r2:.2f}\n"
    )
    
    file_path = "./regressao/results.txt"
    
    # Abrir o arquivo para escrita (cria o arquivo se não existir)
    with open(file_path, 'a') as file:
        file.write(results_str + '\n')


def linear_pipeline(preprocessor, config):
    if config == 1:
        model =  LinearRegression() #configuração padrão
    elif config == 2:
        model = LinearRegression(fit_intercept=False) # sem interceptação, ou seja, a linha de regressão passa pela origem
    elif config == 3:
         model =  LinearRegression(fit_intercept=False, copy_X=False) # sem interceptação e sem copiar os dados de entrada, economizando memória
    elif config == 4:
        model = LinearRegression(copy_X=False) # não copiar os dados de entrada, economizando memória, mas modificando os dados de entrada
    else:
        model = LinearRegression(n_jobs=-1) # usar todos os núcleos do processador para ajustar o modelo, ou seja, paralelizar o processo, portanto, mais rápido
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

def knn_pipeline(preprocessor, config):
    if config == 1:
        model = KNeighborsRegressor(n_neighbors=5) #configuração padrão
    elif config == 2:
        model = KNeighborsRegressor(n_neighbors=10) # aumentar o número de vizinhos
    elif config == 3:
        model = KNeighborsRegressor(n_neighbors=3) # diminuir o número de vizinhos
    elif config == 4:
        model = KNeighborsRegressor(n_neighbors=5, weights='distance') # atribuir pesos aos vizinhos com base na distância
    else:
        model = KNeighborsRegressor(n_neighbors=10, weights='distance') 
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

def tree_pipeline(preprocessor, config):
    if config == 1:
        model = DecisionTreeRegressor() #configuração padrão
    elif config == 2:
        model = DecisionTreeRegressor(max_depth=3) # limitar a profundidade da árvore
    elif config == 3:
        model = DecisionTreeRegressor(min_samples_split=5) # definir o número mínimo de amostras necessárias para dividir um nó
    elif config == 4:
        model = DecisionTreeRegressor(min_samples_leaf=5) # definir o número mínimo de amostras necessárias para ser uma folha
    else:
        model = DecisionTreeRegressor(max_features='sqrt') # número máximo de recursos a serem considerados para dividir um nó
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

def mlp_pipeline(preprocessor, config):
    if config == 1:
        model = MLPRegressor(hidden_layer_sizes=(7), activation='logistic', max_iter=500, random_state=1, alpha=0.01, momentum=0.9) #configuração padrão 
    elif config == 2:
        model = MLPRegressor(hidden_layer_sizes=(7, 7), activation='relu', max_iter=1000, random_state=1, alpha=0.001, momentum=0.8) # duas camadas ocultas com 7 neurônios cada e função de ativação ReLU, aumentar o número de iterações para garantir a convergência, reduzir o parâmetro alpha para regularização L2 e aumentar o momentum para acelerar a convergência
    elif config == 3:
        model = MLPRegressor(hidden_layer_sizes=(7), activation='logistic', max_iter=500, random_state=1, alpha=0.001, momentum=0.9) # aumentar o parâmetro alpha para regularização L2, reduzindo o overfitting, e aumentar o momentum para acelerar a convergência
    elif config == 4:
        model = MLPRegressor(hidden_layer_sizes=(7), activation='logistic', max_iter=500, random_state=1, alpha=0.01, momentum=0.8) # reduzir o parâmetro alpha para regularização L2, aumentando a complexidade do modelo, e reduzir o momentum para uma convergência mais lenta
    else:
        model = MLPRegressor(hidden_layer_sizes=(9), activation='logistic', max_iter=100, random_state=1, alpha=0.01, momentum=0.9) # aumentar o número de neurônios na camada oculta, aumentando a complexidade do modelo, reduzir o número de iterações para economizar tempo e manter os outros hiperparâmetros
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

def main(): 

    # Pré-processamento
    X, y, preprocessor = preprocess(pd.read_csv('./regressao/GallusGallusDomesticus.csv'))

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Treinar e avaliar os modelos
    for config in range(1, 2): 
        models = {
            'Regressao Linear': linear_pipeline(preprocessor,config),
            'KNN Regressor': knn_pipeline(preprocessor, config),
            'Arvore de Decisao Regressora': tree_pipeline(preprocessor, config),
            'MLP Regressor': mlp_pipeline(preprocessor, config)
        }
        
        for name, model in models.items():
            mse, rmse, mae, r2, y_pred = model_evaluation(model, X_train, y_train, X_test, y_test)
            #score = cross_val_score(model, X_train, y_train, cv=10)
            print_results(name, mse, rmse, mae, r2)
            #print(scores)
            plot_results(y_test, y_pred, name, config)

main()