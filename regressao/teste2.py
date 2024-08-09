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
# Pré-processamento
df = pd.read_csv('./regressao/ChickWeight.csv')

print(df.columns)
print(df.drop('weight', axis=1).columns)	