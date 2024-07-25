import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# Importing the dataset
dataset = pd.read_csv('../../Datasets/50_Startups.csv')
X = dataset.iloc[:, :-1]  # Independent Variables
y = dataset.iloc[:, 4]  # Dependent Variable
print(X.head())

# Convert the Categorical column (State)
# If there is 2 categories in a column, we can ue LabelEncoding, otherwise, use OneHotEncoding
states = pd.get_dummies(X['State'], drop_first=True)
print(states.head())

# Drop the state column
X = X.drop('State', axis=1)
X = pd.concat([X, states], axis=1)
print(X.head())

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Predicting the Test set results
y_prediction = linear_regression.predict(X_test)

score = r2_score(y_test, y_prediction)
print('R2 score:', score)

print('New Prediction:', linear_regression.predict([[50298.13, 101530.06, 503876.68, False, True]]))
print()

# Using LabelEncoder and OneHotEncoder
# Importing the dataset
dataset = pd.read_csv('../../Datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values  # Independent Variables
y = dataset.iloc[:, 4].values  # Dependent Variable

# Convert the Categorical column (State)
label_encoder_x = LabelEncoder()
X[:, 3] = label_encoder_x.fit_transform(X[:, 3])
one_hot_encoder = ColumnTransformer([('State', OneHotEncoder(), [3])], remainder='passthrough')
X = one_hot_encoder.fit_transform(X)
# print(X)

# Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Predicting the Test set results
y_prediction = linear_regression.predict(X_test)

print('Train Score: ', linear_regression.score(X_train, y_train))
print('Test Score: ', linear_regression.score(X_test, y_test))
score = r2_score(y_test, y_prediction)
print('R2 score:', score)
