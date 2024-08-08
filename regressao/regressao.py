import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Carregar o dataset
df = pd.read_csv('./regressao/GallusGallusDomesticus.csv')

# Selecionar colunas relevantes para regressão
df = df[['Day', 'Age', 'GallusWeight', 'GallusEggWeight', 'AmountOfFeed', 'SunLightExposure']]

print(df.head())

# Exibir informações gerais sobre o dataset
print(df.info())

# Exibir estatísticas descritivas para variáveis numéricas
print(df.describe())

# Exibir a quantidade de valores nulos por coluna
print(df.isnull().sum())

# Definir recursos (features) e alvo (target)
X = df.drop('GallusWeight', axis=1)
y = df['GallusWeight']

# Identificar colunas numéricas
numeric_features = ['Day', 'Age', 'GallusEggWeight', 'AmountOfFeed', 'SunLightExposure']

# Definir o ColumnTransformer para aplicar transformações às colunas numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

# Definir o pipeline para Regressão Linear
linear_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
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