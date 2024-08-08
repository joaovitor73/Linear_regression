import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc

# Carregar a base de dados Titanic
df = pd.read_csv('titanic_train.csv')

# Selecionar colunas relevantes para classificação binária
df = df[['age', 'fare', 'embarked', 'sex', 'pclass', 'survived']]

# Definir recursos (features) e alvo (target)
X = df.drop('survived', axis=1)
y = df['survived']

# Identificar colunas numéricas e categóricas
numeric_features = ['age', 'fare']
categorical_features = ['embarked', 'sex', 'pclass']

# Criar transformers para colunas numéricas e categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Criar o pré-processador usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Definir os pipelines para os modelos

# K-Nearest Neighbors
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))  # Número de vizinhos
])

# Árvore de Decisão
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# MLP Classifier
mlp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(7),  # Duas camadas ocultas com 50 neurônios cada
        activation='logistic',            # Função de ativação ReLU
        solver='adam',                # Otimizador Adam
        max_iter=500,                 # Número máximo de iterações
        random_state=1,
        alpha=0.01,
		momentum=0.9
    ))
])

# Naive Bayes Multinomial
nb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultinomialNB())
])

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Treinar os modelos e gerar previsões de probabilidade
pipelines = {
    'K-Nearest Neighbors': knn_pipeline,
    'Árvore de Decisão': tree_pipeline,
    'MLP Classifier': mlp_pipeline,
    'Naive Bayes Multinomial': nb_pipeline
}

plt.figure(figsize=(14, 10))

for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Linha Aleatória')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC para Classificadores')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
