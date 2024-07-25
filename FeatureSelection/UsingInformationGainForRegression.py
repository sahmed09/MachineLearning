import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, SelectPercentile

"""Feature Selection-Information gain - mutual information In Regression Problem Statements
Mutual Information (MI)
Estimate mutual information for a continuous target variable.
MI between two random variables is a non-negative value, which measures the dependency between the variables. It is 
equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
The function relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances

Mutual information is calculated between two variables and measures the reduction in uncertainty for one variable 
given a known value of the other variable.

A quantity called mutual information measures the amount of information one can obtain from one random variable given 
another. The mutual information between two random variables X and Y can be stated formally as follows:
I(X ; Y) = H(X) – H(X | Y) Where I(X ; Y) is the mutual information for X and Y, H(X) is the entropy for X 
and H(X | Y) is the conditional entropy for X given Y. The result has the units of bits."""

"""Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data"""

housing_df = pd.read_csv('../Datasets/house-price-train.csv')
print(housing_df.head())
print(housing_df.info())
print(housing_df.isnull().sum())

numeric_lst = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_cols = list(housing_df.select_dtypes(include=numeric_lst).columns)
print(numerical_cols)

housing_df = housing_df[numerical_cols]
print(housing_df.head())

housing_df = housing_df.drop('Id', axis=1)

X = housing_df.drop(labels=['SalePrice'], axis=1)
y = housing_df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)
print(X_train.isnull().sum())

# determine the mutual information
mutual_info = mutual_info_regression(X_train.fillna(0), y_train)
print(mutual_info)

mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
print(mutual_info.sort_values(ascending=False))

mutual_info.sort_values(ascending=False).plot.bar(figsize=(15, 7))
plt.show()

# Selecting the top 20 percentile
selected_top_columns = SelectPercentile(mutual_info_regression, percentile=20)
selected_top_columns.fit(X_train.fillna(0), y_train)
print(selected_top_columns.get_support())

features = X_train.columns[selected_top_columns.get_support()]
print(features)

"""Difference Between Information Gain And Mutual Information
I(X ; Y) = H(X) – H(X | Y) and IG(S, a) = H(S) – H(S | a)
As such, mutual information is sometimes used as a synonym for information gain. Technically, they calculate the 
same quantity if applied to the same data."""
