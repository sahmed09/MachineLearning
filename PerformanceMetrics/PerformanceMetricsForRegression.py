import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""California Housing Dataset"""
print('California Housing Dataset')
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

df['target'] = data.target
print(df.head())

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)
adjusted_r_squared = 1 - ((1 - r_squared) * (len(y_test) - 1)) / (len(y_test) - X_test.shape[1] - 1)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R squared:", r_squared)
print("Adjusted R squared:", adjusted_r_squared)

"""Diabetes Dataset"""
print('\nDiabetes Dataset')
df = load_diabetes()
X = pd.DataFrame(df['data'], columns=df['feature_names'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)
adjusted_r_squared = 1 - ((1 - r_squared) * (len(y_test) - 1)) / (len(y_test) - X_test.shape[1] - 1)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R squared:", r_squared)
print("Adjusted R squared:", adjusted_r_squared)
