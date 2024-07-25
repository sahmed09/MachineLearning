import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

data = [['Male', 23, 24, 'Yes'], [np.NAN, 24, 25, 'No'], ['Female', 25, 26, 'Yes'], ['Male', 26, 27, 'Yes']]
df = pd.DataFrame(data, columns=['F1', 'F2', 'F3', 'F4'])
print(df)
print(df.isnull().sum())

"""Strategy-1 (Delete the missing observations)"""
print('\nStrategy-1 (Delete the missing observations)')
df.dropna(axis=0, inplace=True)
print(df)

"""Strategy-2 (Replace missing values with the most frequent value)"""
print('\nStrategy-2 (Replace missing values with the most frequent value)')
data = [['Male', 23, 24, 'Yes'], [np.NAN, 24, 25, 'No'], ['Female', 25, 26, 'Yes'], ['Male', 26, 27, 'Yes']]
df = pd.DataFrame(data, columns=['F1', 'F2', 'F3', 'F4'])
df = df.fillna(df.mode().iloc[0])
print(df)

"""Strategy-3 (Delete the variable which is having missing values)"""
print('\n Strategy-3 (Delete the variable which is having missing values)')
data = [['Male', 23, 24, 'Yes'], [np.NAN, 24, 25, 'No'], ['Female', 25, 26, 'Yes'], ['Male', 26, 27, 'Yes']]
df = pd.DataFrame(data, columns=['F1', 'F2', 'F3', 'F4'])
df.dropna(axis=1, inplace=True)
print(df)

"""Strategy-4 (Develop a model to predict missing values)"""
print('\nStrategy-4 (Develop a model to predict missing values)')
data = [['Male', 23, 24, 'Yes'], [np.NAN, 24, 25, 'No'], ['Female', 25, 26, 'Yes'], ['Male', 26, 27, 'Yes']]
df = pd.DataFrame(data, columns=['F1', 'F2', 'F3', 'F4'])

encoded_data = pd.get_dummies(df, columns=['F4'], drop_first=True)
print(encoded_data)

test = encoded_data.loc[1:1, ['F1', 'F2', 'F3', 'F4_Yes']]
print(test)

train = encoded_data.dropna(axis=0)
print(train)

train['F1'] = train['F1'].map({'Male': 1, 'Female': 0})
X_train = train.iloc[:, 1:4]
print(X_train)

y_train = train.iloc[:, 0]
print(y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)
inp = [[24, 25, 0]]
print(lr.predict(inp))
