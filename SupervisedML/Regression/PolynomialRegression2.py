import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data_set = pd.read_csv('../../Datasets/Position_Salaries.csv')
print(data_set)

# Extracting Independent and dependent Variable
X = data_set.iloc[:, 1:2].values
y = data_set.iloc[:, 2].values
print(X)
print(y)

# # Fitting the Linear Regression to the dataset
linear_regression = LinearRegression()
linear_regression.fit(X, y)

# Fitting the Polynomial regression to the dataset
poly_regression_2 = PolynomialFeatures(degree=2)
X_poly = poly_regression_2.fit_transform(X)
linear_regression_2 = LinearRegression()
linear_regression_2.fit(X_poly, y)

poly_regression_3 = PolynomialFeatures(degree=3)
X_poly = poly_regression_3.fit_transform(X)
linear_regression_3 = LinearRegression()
linear_regression_3.fit(X_poly, y)

# Visualizing the result for Linear Regression model
plt.scatter(X, y, color='blue')
plt.plot(X, linear_regression.predict(X), color='red')
plt.title("Bluff detection model(Linear Regression)")
plt.xlabel("Position Levels")
plt.ylabel('Salary')
plt.show()

# Visualizing the result for Polynomial Regression
plt.scatter(X, y, color='blue')
plt.plot(X, linear_regression_2.predict(poly_regression_2.fit_transform(X)), color='red')
plt.plot(X, linear_regression_3.predict(poly_regression_3.fit_transform(X)), color='yellow')
plt.title("Bluff detection model(Polynomial Regression)")
plt.xlabel("Position Levels")
plt.ylabel('Salary')
plt.show()

lin_prediction = linear_regression.predict([[6.5]])
print('Linear Regression Prediction:', lin_prediction)

poly_prediction = linear_regression_2.predict(poly_regression_2.fit_transform([[6.5]]))
print('polynomial Regression Prediction (Degree 2):', poly_prediction)

poly_prediction = linear_regression_3.predict(poly_regression_3.fit_transform([[6.5]]))
print('polynomial Regression Prediction (Degree 3):', poly_prediction)
