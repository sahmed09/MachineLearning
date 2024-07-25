import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""One Hot Encoding"""
print('One Hot Encoding')

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Sex'])
print(df.head())
print(pd.get_dummies(df, drop_first=True).head())  # One Hot Encoding

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Embarked'])
print(df.head())
print(df['Embarked'].unique())

df.dropna(inplace=True)
print(df['Embarked'].unique())

print(pd.get_dummies(df, drop_first=True).head(10))  # One Hot Encoding

# Onehotencoding with many categories in a feature
df = pd.read_csv('../Datasets/mercedesbenz.csv', usecols=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6'])
print(df.head())
print(df.shape)

print(df['X0'].value_counts())
print(df['X0'].unique())

for col in df.columns:
    print(col, ':', len(df[col].unique()), 'labels')

print(df.X1.value_counts().sort_values(ascending=False).head(10))

top_10_features_X1 = df.X1.value_counts().sort_values(ascending=False).head(10).index
print(top_10_features_X1)
top_10_features_X1 = list(top_10_features_X1)
print('Top 10 Categories in X1:', top_10_features_X1)

for categories in top_10_features_X1:
    df[categories] = np.where(df['X1'] == categories, 1, 0)

top_10_features_X1.append('X1')
print(df[top_10_features_X1])

"""Ordinal Number Encoding or Label Encoding"""
print('\nOrdinal Number Encoding or Label Encoding')

today_date = datetime.datetime.today()
print('Today:', today_date)
print(today_date - datetime.timedelta(3))

# List Comprehension of last 15 days
days = [today_date - datetime.timedelta(x) for x in range(0, 15)]

data = pd.DataFrame(days)
data.columns = ['day']
print(data.head())

data['weekday'] = data['day'].dt.day_name()
print(data.head())

dictionary = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
print(dictionary)

data['weekday_ordinal'] = data['weekday'].map(dictionary)
print(data.head())

"""Count Or Frequency Encoding"""
print('\nCount Or Frequency Encoding')
# Dataset Link: http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

train_set = pd.read_csv('../Datasets/adult.data', header=None, index_col=None)
print(train_set.head())
print(train_set.shape)

columns = [1, 3, 5, 6, 7, 8, 9, 13]
train_set = train_set[columns]
train_set.columns = ['Employment', 'Degree', 'Status', 'Designation', 'family_job', 'Race', 'Sex', 'Country']
print(train_set.head())

for feature in train_set.columns[:]:
    print(feature, ':', len(train_set[feature].unique()), 'labels')

country_map = train_set['Country'].value_counts().to_dict()
print(country_map)

train_set['Country'] = train_set['Country'].map(country_map)
print(train_set.head(10))

"""Target Guided Ordinal Encoding"""
print('\nTarget Guided Ordinal Encoding')

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Cabin', 'Survived'])
print(df.head())

df['Cabin'].fillna('Missing', inplace=True)
print(df.head())

df['Cabin'] = df['Cabin'].astype(str).str[0]
print(df.head())

print('Cabin Unique Categories:', df.Cabin.unique())

print(df.groupby(['Cabin'])['Survived'].mean())
print(df.groupby(['Cabin'])['Survived'].mean().sort_values().index)

ordinal_labels = df.groupby(['Cabin'])['Survived'].mean().sort_values().index
print(ordinal_labels)

print(enumerate(ordinal_labels, 0))

ordinal_labels2 = {k: i for i, k in enumerate(ordinal_labels, 0)}
print(ordinal_labels2)

df['Cabin_ordinal_labels'] = df['Cabin'].map(ordinal_labels2)
print(df.head())

"""Mean Encoding"""
print('\nMean Encoding')

mean_ordinal = df.groupby(['Cabin'])['Survived'].mean().to_dict()
print(mean_ordinal)

df['mean_ordinal_encode'] = df['Cabin'].map(mean_ordinal)
print(df.head())

"""Probability Ratio Encoding"""
print('\nProbability Ratio Encoding')

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Cabin', 'Survived'])
print(df.head())

# Replacing
df['Cabin'].fillna('Missing', inplace=True)
print(df.head())

print('Cabin Unique Categories:', df['Cabin'].unique())
print('Length of Cabin Unique Categories:', len(df['Cabin'].unique()))

df['Cabin'] = df['Cabin'].astype(str).str[0]
print(df.head())

print('Cabin Unique Categories:', df['Cabin'].unique())
print('Length of Cabin Unique Categories:', len(df['Cabin'].unique()))

# Step-1: Probability of Survived based on Cabin
prob_df = df.groupby(['Cabin'])['Survived'].mean()
prob_df = pd.DataFrame(prob_df)
print(prob_df)

# Step-2: Probability of Not Survived
prob_df['Died'] = 1 - prob_df['Survived']
print(prob_df.head())

# Step-3: pr(Survived)/pr(Not Survived)
prob_df['Probability_ratio'] = prob_df['Survived'] / prob_df['Died']
print(prob_df.head())

# Step-4: Dictionary to map cabin with probability
probability_encoded = prob_df['Probability_ratio'].to_dict()
print(probability_encoded)

# Step-5: replace with the categorical feature
df['Cabin_encoded'] = df['Cabin'].map(probability_encoded)
print(df.head())
