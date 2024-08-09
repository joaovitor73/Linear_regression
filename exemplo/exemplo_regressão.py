import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Carregar a base de dados Titanic
df = pd.read_csv('./exemplo/titanic_test.csv')

# Selecionar colunas relevantes para regressão
df = df[['age', 'fare', 'embarked', 'sex', 'pclass']]

# Definir recursos (features) e alvo (target)
X = df.drop('fare', axis=1)
y = df['fare']

# Identificar colunas numéricas e categóricas
numeric_features = ['age']
categorical_features = ['embarked', 'sex', 'pclass']

# Criar transformers para colunas numéricas e categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 
])

# Criar o pré-processador usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Definir o pipeline para Regressão Linear
linear_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# K-Nearest Neighbors Regressor
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=5))  # Número de vizinhos
])

# Árvore de Decisão Regressora
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

# MLP Regressor
mlp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(
        hidden_layer_sizes=(7),  # Duas camadas ocultas com 50 neurônios cada
        activation='logistic',            # Função de ativação ReLU
        max_iter=500,                 # Número máximo de iterações
        random_state=1,
		alpha=0.01,
		momentum=0.9
    ))
])



# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Treinar o modelo de Regressão Linear
linear_pipeline.fit(X_train, y_train)

# Fazer previsões
y_pred = linear_pipeline.predict(X_test)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Regressão Linear:")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
print(f"Erro Absoluto Médio (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Criar e exibir o gráfico de dispersão
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Regressão Linear: Valores Reais vs. Preditos')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)  # Linha de referência
plt.grid(True)
plt.show()
