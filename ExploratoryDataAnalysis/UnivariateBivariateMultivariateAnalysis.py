import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition

# df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
df = sns.load_dataset('iris')
print(df.head())
print(df.shape)
print(df['species'].unique())
print(df.dtypes)
print(df['species'].value_counts())

"""Univariate Analysis"""
print('Univariate Analysis')

df_setosa = df.loc[df['species'] == 'setosa']
print(df_setosa)
df_virginica = df.loc[df['species'] == 'virginica']
df_versicolor = df.loc[df['species'] == 'versicolor']

plt.plot(df_setosa['sepal_length'], np.zeros_like(df_setosa['sepal_length']), 'o')
plt.plot(df_virginica['sepal_length'], np.zeros_like(df_virginica['sepal_length']), 'o')
plt.plot(df_versicolor['sepal_length'], np.zeros_like(df_versicolor['sepal_length']), 'o')
plt.xlabel('Patel Length')
plt.title('Univariate Analysis')
plt.show()

"""Bivariate Analysis"""
print('Bivariate Analysis')
sns.FacetGrid(df, hue='species', height=5, aspect=1).map(plt.scatter, 'petal_length', 'sepal_width').add_legend()
plt.show()

"""Multivariate Analysis"""
print('Multivariate Analysis')
sns.pairplot(df, hue='species', height=3)
plt.show()

"""Univariate Analysis"""
print('Univariate Analysis')

data = pd.read_csv('../Datasets/Employee_dataset.csv')
print(data.head())
print(data.columns)

# performing univariate analysis on Numerical variables using the histogram function
sns.histplot(data['age'])
plt.title('Age Details')
plt.show()

# Univariate analysis of categorical data using the count plot function (Bar Chart)
sns.countplot(data, x='gender_full', hue='gender_full')
plt.title('Gender Details')
plt.show()

# A piechart helps us to visualize the percentage of the data belonging to each category
x = data['STATUS_YEAR'].value_counts()
print(x)
plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
plt.show()

"""Bivariate Analysis"""
print('Bivariate Analysis')

# Numerical v/s Numerical
sns.scatterplot(x=data['length_of_service'], y=data['age'])
plt.title('Numerical v/s Numerical Analysis')
plt.show()

# Numerical v/s Categorical
plt.figure(figsize=(15, 5))
sns.barplot(x=data['department_name'], y=data['length_of_service'], hue=data['department_name'])
plt.xticks(rotation='vertical')
plt.title('Numerical v/s Categorical Analysis')
plt.show()
# The Black horizontal line is indicating huge differences in the length of service among different departments.

# Categorical V/s Categorical
sns.countplot(x=data['STATUS_YEAR'], hue=data['STATUS'])
plt.title('Categorical V/s Categorical Analysis')
plt.legend(loc='upper left')
plt.show()

"""Multivariate Analysis"""
print('Multivariate Analysis')

iris = datasets.load_iris()

X = iris.data
y = iris.target

pca = decomposition.PCA(n_components=2)
X = pca.fit_transform(X)
# print(X)

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
plt.show()

print(data.dtypes)
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.show()
