import numpy as np
import pandas as pd
import xgboost
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error

"""XGBoost Classifier (Customers Details Dataset)"""
print('XGBoost Classifier (Customers Details Dataset)')
customer_dataset = pd.read_csv('../../Datasets/Churn_Modelling.csv')
print(customer_dataset.head())
# print(customer_dataset.info())

X = customer_dataset.iloc[:, 3:13]
y = customer_dataset.iloc[:, 13]

# Define categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# print(categorical_features)
# print(numerical_features)

# Define preprocessing steps for categorical and numerical features
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features),
    ('num', StandardScaler(), numerical_features),
])
# print(preprocessor)
X = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

my_model = XGBClassifier()
my_model.fit(X_train, y_train)
y_pred = my_model.predict(X_test)

c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)

print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Hyper-tuning XGBoost using Grid Search method
parameters = {'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}
xgb_model = XGBClassifier()
grid_search = GridSearchCV(estimator=xgb_model, param_grid=parameters, cv=10, scoring='roc_auc', n_jobs=-1,
                           verbose=True)
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print('Best Parameters:', best_params)
# print('Best Model:', best_model)
print('Best Score:', grid_search.best_score_)

y_predictions_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_predictions_best)
precision_best = precision_score(y_test, y_predictions_best)
recall_best = recall_score(y_test, y_predictions_best)
f1_best = f1_score(y_test, y_predictions_best)
print(f'Best Model Accuracy (GridSearchCV): {accuracy_best * 100}')
print(f'Best Model Precision (GridSearchCV): {precision_best}')
print(f'Best Model Recall (GridSearchCV): {recall_best}')
print(f'Best Model F1 Score (GridSearchCV): {f1_best}')

"""XGBoost Classifier (Customers Details Dataset - 2nd Approach)"""
print('\nXGBoost Classifier (Customers Details Dataset - 2nd Approach)')
customer_dataset = pd.read_csv('../../Datasets/Churn_Modelling.csv')

label_encoder = LabelEncoder()
customer_dataset['Geography'] = label_encoder.fit_transform(customer_dataset['Geography'])
customer_dataset['Gender'] = label_encoder.fit_transform(customer_dataset['Gender'])
# print(customer_dataset.info())

X = customer_dataset.iloc[:, 3:13].values
y = customer_dataset.iloc[:, 13].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

xgb_train = xgboost.DMatrix(X_train, y_train, enable_categorical=True)
xgb_test = xgboost.DMatrix(X_test, y_test, enable_categorical=True)

params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1}

model = xgboost.train(params=params, dtrain=xgb_train, num_boost_round=50)
preds = model.predict(xgb_test)
preds = preds.astype(int)
accuracy = accuracy_score(y_test, preds)
print('Accuracy of the model is:', accuracy * 100)

# Hyper-tuning XGBoost using Grid Search method
parameters = {'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}
xgb_model = XGBClassifier()
grid_search = GridSearchCV(estimator=xgb_model, param_grid=parameters, cv=10, scoring='roc_auc', n_jobs=-1,
                           verbose=True)
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print('Best Parameters:', best_params)
# print('Best Model:', best_model)
print('Best Score:', grid_search.best_score_)

y_predictions_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_predictions_best)
precision_best = precision_score(y_test, y_predictions_best)
recall_best = recall_score(y_test, y_predictions_best)
f1_best = f1_score(y_test, y_predictions_best)
print(f'Best Model Accuracy (GridSearchCV): {accuracy_best * 100}')
print(f'Best Model Precision (GridSearchCV): {precision_best}')
print(f'Best Model Recall (GridSearchCV): {recall_best}')
print(f'Best Model F1 Score (GridSearchCV): {f1_best}')

"""XGBoost Classifier (Customers Details Dataset - Hyperparameter optimization using RandomizedSearchCV)"""
print('\nXGBoost Classifier (Customers Details Dataset - Hyperparameter optimization using RandomizedSearchCV)')
customer_dataset = pd.read_csv('../../Datasets/Churn_Modelling.csv')
print(customer_dataset.head())

# get correlations of each features in dataset
corr_mat = customer_dataset.corr(numeric_only=True)
top_corr_features = corr_mat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(customer_dataset[top_corr_features].corr(), annot=True, cmap='RdYlGn')
plt.show()

X = customer_dataset.iloc[:, 3:13]
y = customer_dataset.iloc[:, 13]

geography = pd.get_dummies(X['Geography'], drop_first=True)
print(geography.head())

gender = pd.get_dummies(X['Gender'], drop_first=True)
print(gender.head())

# Drop Categorical Features
X = X.drop(['Geography', 'Gender'], axis=1)
print(X.head())

X = pd.concat([X, geography, gender], axis=1)
print(X.head())


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\nTime taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# Hyperparameter optimization using RandomizedSearchCV
params = {'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30], 'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
          'min_child_weight': [1, 3, 5, 7], 'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
          'colsample_bytree': [0.3, 0.4, 0.5, 0.07]}

classifier = xgboost.XGBClassifier()
random_search = RandomizedSearchCV(estimator=classifier, param_distributions=params, n_iter=5, scoring='roc_auc',
                                   n_jobs=-1, cv=5, verbose=3)

start_time = timer(None)  # timing starts from this point for 'start_time' variable
random_search.fit(X, y)
timer(start_time)  # timing ends here for 'start_time' variable

# print(random_search.best_estimator_)
print(random_search.best_params_)

best_estimator = random_search.best_estimator_
best_params = best_estimator.get_params()
print(best_params)

new_classifier = xgboost.XGBClassifier(**params)
score = cross_val_score(classifier, X, y, cv=10)
print('Scores:', score)
print('Mean Score:', score.mean())

"""XGBoost Regressor (California Dataset)"""
print('\nXGBoost Regressor (California Dataset)')
california = fetch_california_housing()
X = california.data
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10,
                      n_estimators=10)
xg_reg.fit(X_train, y_train)
y_preds = xg_reg.predict(X_test)

mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
