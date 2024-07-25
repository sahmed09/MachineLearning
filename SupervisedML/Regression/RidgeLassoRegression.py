import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]
# print(data)
# print(target)

df = fetch_california_housing()
print(df)

dataset = pd.DataFrame(df.data)
print(dataset.head())

dataset.columns = df.feature_names
print(dataset.head())
print(dataset.shape)
print(dataset.columns)
print(df.target.shape)

dataset['Price'] = df.target
print(dataset.head())

X = dataset.iloc[:, :-1]  # independent features
y = dataset.iloc[:, -1]  # dependent features
print(X.shape, y.shape)

"""Linear Regression"""
liner_regression = LinearRegression()
mse = cross_val_score(liner_regression, X, y, scoring='neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)
print('Mean MSE:', mean_mse)

"""Ridge Regression"""
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 75, 85, 100]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X, y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

"""Lasso Regression"""
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 75, 85, 100]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X, y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
prediction_lasso = lasso_regressor.predict(X_test)
prediction_ridge = ridge_regressor.predict(X_test)

sns.displot(y_test - prediction_lasso)
plt.show()

sns.displot(y_test - prediction_ridge)
plt.show()
