import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

"""Implement Random Forest for Classification -> Titanic Dataset"""
titanic_data = pd.read_csv('../../Datasets/titanic_train.csv')
print(titanic_data.shape)
print(titanic_data.head())
print(titanic_data.isnull().sum())

# Drop rows with missing target values
titanic_data = titanic_data.dropna(subset=['Survived'])
print(titanic_data.shape)

# Select relevant features and target variable
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

# Convert categorical variable 'Sex' to numerical using .loc
X.loc[:, 'Sex'] = X['Sex'].map({'female': 0, 'male': 1})

# Handle missing values in the 'Age' column using .loc
X.loc[:, 'Age'].fillna(X['Age'].median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

"""Implement Random Forest for Classification -> Titanic Dataset (2nd Approach)"""
titanic_data = pd.read_csv('../../Datasets/titanic_train.csv')
print(titanic_data.shape)
print(titanic_data.head())
print(titanic_data.isnull().sum())

# Data preprocessing
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
titanic_data = titanic_data.fillna(0)

# Handling categorical data
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'].astype(str))
titanic_data['Embarked'] = label_encoder.fit_transform(titanic_data['Embarked'].astype(str))
print(titanic_data.head())

# Dependent and independent variables
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print(y_pred)

rand_score = rf_classifier.score(X_test, y_test)
print('Score:', rand_score)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)

report = classification_report(y_test, y_pred)
print(report)

"""Implement Random Forest for Regression -> California Dataset"""
california_housing = fetch_california_housing()
california_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_data['MEDV'] = california_housing.target

X = california_data.drop('MEDV', axis=1)
y = california_data['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

"""Credit Card Dataset"""
df = pd.read_csv('../../Datasets/CreditCard.csv')
print(df.head())

# Transform into Numerical Data and Isolate
df = pd.get_dummies(data=df, drop_first=True)
print(df.head())

X = df.drop(columns=['card_yes'])
y = df.card_yes
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(len(X_train), len(X_test), len(y_train), len(y_test))

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
model_predict = model.predict(X_test)

f1 = f1_score(y_test, model_predict)
precision = precision_score(y_test, model_predict)
recall = recall_score(y_test, model_predict)
print('F1 Score:', f1)
print('Precision:', precision)
print('Recall:', recall)

c_matrix = confusion_matrix(y_test, model_predict)
print(c_matrix)

report = classification_report(y_test, model_predict)
print(report)

# Plotting the Output
importance = pd.Series(model.feature_importances_, index=X_train.columns.values)
importance.nlargest(5).plot(kind='barh', figsize=(6, 4))
plt.show()

# Parameter Tuning Section
param_grid = {'n_estimators': range(50, 500, 50)}
grid = ParameterGrid(param_grid)

f1score = []
for params in grid:
    model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=1502)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)

    f1 = f1_score(y_test, predict)
    f1score.append(f1)

print(grid[np.argmax(f1)])
