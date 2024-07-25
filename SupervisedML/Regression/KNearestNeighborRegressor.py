import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from sklearn.ensemble import BaggingRegressor
import warnings

warnings.filterwarnings('ignore')

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone = pd.read_csv(url, header=None)
print(abalone.head())
# print(abalone.isnull().sum())

abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",
                   "Shell weight", "Rings", ]
print(abalone.head())

abalone = abalone.drop("Sex", axis=1)
# print(abalone.head())

abalone["Rings"].hist(bins=15)
plt.show()

correlation_matrix = abalone.corr()
print(correlation_matrix['Rings'])

X = abalone.drop('Rings', axis=1)
y = abalone['Rings']
print(X.values)
print(y.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
print('Train RMSE:', rmse)

test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
print('Test RMSE:', rmse)

print('Prediction:', knn_model.predict([[0.455, 0.365, 0.101, 0.4567, 0.1875, 0.1125, 0.080]]))
print('Prediction:', knn_model.predict([[0.530, 0.430, 0.145, 0.6920, 0.2895, 0.1245, 0.250]]))

# Plotting the Fit of Your Model
# cmap = sns.cubehelix_palette(as_cmap=True)
# f, ax = plt.subplots()
# points = ax.scatter(X_test[:, 0], X_test[:, 1], c=test_preds, s=50, cmap=cmap)
# f.colorbar(points)
# plt.show()
#
# cmap = sns.cubehelix_palette(as_cmap=True)
# f, ax = plt.subplots()
# points = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=cmap)
# f.colorbar(points)
# plt.show()

# Improving kNN Performances Using GridSearchCV
parameters = {'n_neighbors': range(1, 50)}

gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
print(gridsearch)
print(gridsearch.best_params_)

train_preds_grid = gridsearch.predict(X_train)
train_mse = mean_squared_error(y_train, train_preds_grid)
train_rmse = sqrt(train_mse)
print('Train RMSE (Using GridSearchCV):', train_rmse)

test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print('Test RMSE (Using GridSearchCV):', test_rmse)

# Adding Weighted Average of Neighbors Based on Distance
parameters = {'n_neighbors': range(1, 50), "weights": ["uniform", "distance"], }

gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
print(gridsearch)
print(gridsearch.best_params_)

test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print('Test RMSE (Using GridSearchCV Based on Average Distance):', test_rmse)

# Further Improving on KNN With Bagging
best_k = gridsearch.best_params_["n_neighbors"]
best_weights = gridsearch.best_params_["weights"]
print('best_k:', best_k, 'best_weights:', best_weights)

bagged_knn = KNeighborsRegressor(n_neighbors=best_k, weights=best_weights)

bagging_model = BaggingRegressor(bagged_knn, n_estimators=100)
bagging_model.fit(X_train, y_train)

test_preds_grid = bagging_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print('Test RMSE (Bagging):', test_rmse)
