import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

"""Gradient Boosting Classifier (Breast Cancer Dataset)"""
print('\nGradient Boosting Classifier (Breast Cancer Dataset)')
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# df = pd.DataFrame(load_breast_cancer()['data'], columns=load_breast_cancer()['feature_names'])
# df['y'] = load_breast_cancer()['target']

print('Name of the Features:', breast_cancer.feature_names)
print('Name of the classes:', breast_cancer.target_names)
print('Shape of the classes:', breast_cancer.data.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

GBM = GradientBoostingClassifier(n_estimators=100, random_state=42)
# GBM = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
GBM.fit(X_train, y_train)
y_pred = GBM.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy of Gradient Boosting:', accuracy)
print('Precision of Gradient Boosting:', precision)
print('Recall of Gradient Boosting:', recall)
print('f1 Score of Gradient Boosting:', f1)

c_matrix = confusion_matrix(y_test, y_pred)
c_matrix_df = pd.DataFrame(c_matrix, index=[i for i in range(2)], columns=[i for i in range(2)])
print(c_matrix)
print(c_matrix_df)

report = classification_report(y_test, y_pred)
print(report)

"""Gradient Boosting Classifier (Digits Dataset)"""
print('\nGradient Boosting Classifier (Digits Dataset)')
X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

gbc = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=100, max_features=5)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Gradient Boosting Classifier accuracy is : {:.2f}'.format(accuracy))

"""Gradient Boosting Classifier (Wine Dataset)"""
print('\nGradient Boosting Classifier (Wine Dataset)')
wine = load_wine()

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

GBC = GradientBoostingClassifier()
parameters = {'learning_rate': [0.01, 0.02, 0.03], 'subsample': [0.9, 0.5, 0.2], 'n_estimators': [100, 500, 1000],
              'max_depth': [4, 6, 8]}

# Run through GridSearchCV and print results
grid_GBC = GridSearchCV(estimator=GBC, param_grid=parameters, cv=2, n_jobs=-1)
grid_GBC.fit(X_train, y_train)

print('Results from Grid Search')
print('The best estimator across ALL searched params:', grid_GBC.best_estimator_)
print('The best score across ALL searched params:', grid_GBC.best_score_)
print('The best parameters across ALL searched params:', grid_GBC.best_params_)

"""Gradient Boosting Regressor (California Dataset)"""
print('\nGradient Boosting Regressor (California Dataset)')
california = fetch_california_housing()
# print(california)

X = california.data
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = GradientBoostingRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Score:', mse)
print('R2 Score:', r2)

"""Gradient Boosting Regressor (Diabetes Dataset)"""
print('\nGradient Boosting Regressor (Diabetes Dataset)')
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

# params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 5, 'learning_rate': 0.01, 'loss': 'squared_error'}
params = {'n_estimators': 300, 'max_depth': 1, 'max_features': 5, 'learning_rate': 0.1, 'loss': 'absolute_error',
          'random_state': 23}
# 'loss': {'squared_error', 'absolute_error', 'quantile', 'huber'}

# gradient boosting classifier
gbr = GradientBoostingRegressor(**params)
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)

mse_gbr = mean_squared_error(y_test, gbr_pred)
rmse_gbr = np.sqrt(mse_gbr)
print('The Mean Squared Error (MSE) on test set for Gradient Boosting: {:.4f}'.format(mse_gbr))
print('Root Mean Squared Error (RMSE) on test set for Gradient Boosting: {:.2f}'.format(rmse_gbr))

# adaboost classifier
abr = AdaBoostRegressor()
abr.fit(X_train, y_train)
abr_pred = abr.predict(X_test)

mse_abr = mean_squared_error(y_test, abr_pred)
print('The Mean Squared Error (MSE) on test set for AdaBoost: {:.4f}'.format(mse_abr))
