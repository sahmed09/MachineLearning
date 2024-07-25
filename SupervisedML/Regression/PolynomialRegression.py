import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X ** 2 + 1.5 * X + 2 + np.random.rand(100, 1)  # Quadratic equation used = 0.5x^2 + 1.5x + 2 + outliers

# plt.scatter(X, y, color='g')
# plt.xlabel('X dataset')
# plt.ylabel('y dataset')
# plt.show()

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression
l_regression = LinearRegression()
l_regression.fit(X_train, y_train)
y_prediction = l_regression.predict(X_test)
score = r2_score(y_test, y_prediction)
print("r_2 score:", score)
print("Coefficients:", l_regression.coef_)
print("Intercept:", l_regression.intercept_)
print('New Prediction:', l_regression.predict([[2.00098743]]))
print()

# Let's visualize the model
# plt.plot(X_train, l_regression.predict(X_train), color='r')
# plt.scatter(X_train, y_train)
# plt.xlabel('X Dataset')
# plt.ylabel('y Dataset')
# plt.show()

# Simple Polynomial Regression
# Apply polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=True)  # Degree=2
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# print(X_train_poly)

l_regression = LinearRegression()
l_regression.fit(X_train_poly, y_train)
y_prediction = l_regression.predict(X_test_poly)
score = r2_score(y_test, y_prediction)
print("r_2 score:", score)
print("Coefficients:", l_regression.coef_)
print("Intercept:", l_regression.intercept_)
print('New Prediction:', l_regression.predict(poly.fit_transform([[2.00098743]])))
print()

# plt.scatter(X_train, l_regression.predict(X_train_poly), color='r')
# plt.scatter(X_train, y_train)
# plt.xlabel('X Dataset')
# plt.ylabel('y Dataset')
# plt.show()

poly = PolynomialFeatures(degree=3, include_bias=True)  # Degree=3
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# print(X_train_poly)

l_regression = LinearRegression()
l_regression.fit(X_train_poly, y_train)
y_prediction = l_regression.predict(X_test_poly)
score = r2_score(y_test, y_prediction)
print("r_2 score:", score)
print("Coefficients:", l_regression.coef_)
print("Intercept:", l_regression.intercept_)
print('New Prediction:', l_regression.predict(poly.fit_transform([[2.00098743]])))

# plt.scatter(X_train, l_regression.predict(X_train_poly), color='r')
# plt.scatter(X_train, y_train)
# plt.xlabel('X Dataset')
# plt.ylabel('y Dataset')
# plt.show()

# Prediction of new Data set
X_new = np.linspace(-3, 3, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
# print(X_new_poly)
y_new = l_regression.predict(X_new_poly)

plt.plot(X_new, y_new, 'r-', linewidth=2, label='New Predictions')
plt.plot(X_train, y_train, 'b.', label='Training Points')
plt.plot(X_test, y_test, 'g.', label='Testing Points')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


# Pipeline Concepts
def polynomial_regression(degree):
    X_new = np.linspace(-3, 3, 200).reshape(200, 1)

    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    lin_regression = LinearRegression()
    poly_regression = Pipeline([
        ('poly_features', poly_features),
        ('lin_regression', lin_regression)
    ])

    poly_regression.fit(X_train, y_train)  # Creating polynomial features and fit of linear regression
    y_prediction_new = poly_regression.predict(X_new)

    # Plotting Prediction Line
    plt.plot(X_new, y_prediction_new, 'r', label='Prediction Degree=' + str(degree), linewidth=2)
    plt.plot(X_train, y_train, 'b.', label='Training Points', linewidth=3)
    plt.plot(X_test, y_test, 'g.', label='Testing Points', linewidth=3)
    plt.legend(loc='upper left')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.axis([-4, 4, 0, 10])
    plt.show()


polynomial_regression(3)
