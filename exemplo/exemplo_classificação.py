
# pip install pandas
import pandas
# pip install scikit-learn
import sklearn

# https://www.kaggle.com/c/titanic/data
# Caminho para o arquivo CSV no sistema de arquivos local
path_to_csv = './titanic_train.csv'
# Carregar o CSV em um DataFrame do pandas
titanic = pandas.read_csv(path_to_csv)
# Visualizar as primeiras linhas do DataFrame
print(titanic.head())


# Vamos usar 'age', 'fare', 'embarked', 'sex', 'pclass' como características e 'survived' como alvo
titanic = titanic[['age', 'fare', 'embarked', 'sex', 'pclass', 'survived']]
# Separar as características (features) e a variável alvo (target)
X = titanic.drop('survived', axis=1)
y = titanic['survived']
print(titanic.head())




from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import MultinomialNB


# Definir colunas numéricas e categóricas
numeric_features = ['age', 'fare']
categorical_features = ['embarked', 'sex', 'pclass']

# Criar transformers para colunas numéricas e categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
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

# Criar um pipeline que inclui o pré-processamento e o classificador k-NN
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

# Definir a métrica de avaliação
scoring = 'accuracy'

# Realizar validação cruzada para o k-NN
knn_scores = cross_val_score(knn_pipeline, X, y, cv=10, scoring=scoring)

print(f"k-NN Cross-Validation Accuracy Scores: {knn_scores}")
print(f"k-NN Cross-Validation Accuracy Mean: {knn_scores.mean()}")
print(f"k-NN Cross-Validation Accuracy Std: {knn_scores.std()}")







# Criar um pipeline que inclui o pré-processamento e o classificador Naive Bayes
nb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])


# Criar um pipeline que inclui o pré-processamento e o classificador MLP
mlp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(7,), max_iter=500, random_state=1, activation='logistic', alpha=0.01, momentum=0.9))
])








# Criar transformers para colunas numéricas e categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('discretizer', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'))  # Discretizar atributos numéricos
])

# Criar o pré-processador usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Criar um pipeline que inclui o pré-processamento e o classificador Árvore de Decisão
decision_tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(criterion='entropy', random_state=1))
])

# Criar um pipeline que inclui o pré-processamento e o classificador Naive Bayes Multinomial
nb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultinomialNB())
])




# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
nb_pipeline.fit(X, y)

# Fazer previsões no conjunto de teste
y_pred = nb_pipeline.predict(X_test)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Relatório de classificação para mais detalhes
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))



# pip install matplotlib
# pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

# Mapear índices para os nomes das classes
class_names = ['Not Survived', 'Survived']
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.title('Matriz de Confusão')
plt.show()
