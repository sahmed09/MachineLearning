import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('Datasets/wine.csv', usecols=[1, 2, 3])
df.columns = ['Class', 'Alcohol', 'Malic']
print(df.head())

"""Normalization"""
scaling = MinMaxScaler()
normalized_value = scaling.fit_transform(df[['Alcohol', 'Malic']])
print(normalized_value)

"""Standardization"""
scaling = StandardScaler()
scaled_values = scaling.fit_transform(df[['Alcohol', 'Malic']])
print(scaled_values)
