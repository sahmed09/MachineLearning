import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2

"""Fisher Score - Chisquare Test For Feature Selection
Compute chi-squared stats between each non-negative feature and class.
This score should be used to evaluate categorical variables in a classification task.
This score can be used to select the n_features features with the highest values for the test chi-squared 
statistic from X, which must contain only non-negative features such as booleans or frequencies (e.g., term counts 
in document classification), relative to the classes.

Recall that the chi-square test measures dependence between stochastic variables, so using this function “weeds out” 
the features that are the most likely to be independent of class and therefore irrelevant for classification. 
The Chi Square statistic is commonly used for testing relationships between categorical variables.

It compares the observed distribution of the different classes of target Y among the different categories of the 
feature, against the expected distribution of the target classes, regardless of the feature categories."""

df = sns.load_dataset('titanic')
print(df.head())
print(df.info())

df = df[['sex', 'embarked', 'alone', 'pclass', 'survived']]
print(df.head())

# Let's perform label encoding on sex column
df['sex'] = np.where(df['sex'] == 'male', 1, 0)
print(df.head())

# let's perform label encoding on embarked
ordinal_label = {k: i for i, k in enumerate(df['embarked'].unique(), 0)}
print(ordinal_label)
df['embarked'] = df['embarked'].map(ordinal_label)
print(df.head())

# let's perform label encoding on alone
df['alone'] = np.where(df['alone'] == True, 1, 0)
print(df.head())

# train Test split is usually done to avoid overfitting
X = df.drop(labels=['survived'], axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(X_train['sex'].unique())
print(X_train.isnull().sum())

# Perform chi2 test
# chi2 returns 2 values (Fscore and the pvalue)
f_score, p_values = chi2(X_train, y_train)
print(f_score, p_values)

p_values = pd.Series(p_values)
p_values.index = X_train.columns
print(p_values)
print(p_values.sort_values(ascending=False))

# f_p_values = chi2(X_train, y_train)
# p_values = pd.Series(f_p_values[1])

"""Observation
Alone Column is the most important column when compared to the output feature Survived"""
