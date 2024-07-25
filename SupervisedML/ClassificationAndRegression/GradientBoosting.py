import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

"""Gradient Boosting Classifier (Titanic Dataset)"""
print('Gradient Boosting Classifier (Titanic Dataset)')
titanic_data = pd.read_csv('../../Datasets/titanic_train.csv')
print(titanic_data.head())
# print(titanic_data.dtypes)

# Replace missing values and encode categorical variables
titanic_data.fillna(0, inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

X = titanic_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scalar = MinMaxScaler()
# X_train = scalar.fit_transform(X_train)
# X_test = scalar.transform(X_test)

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
y_predictions = gb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
print(f'Accuracy (Default GradientBoosting): {accuracy}')

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
training, testing = [], []
for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2,
                                        random_state=0)
    gb_clf.fit(X_train, y_train)
    training.append(gb_clf.score(X_train, y_train))
    testing.append(gb_clf.score(X_test, y_test))
    # print('Learning Rate:', learning_rate)
    # print('Accuracy Score (Training): {0:.3f}'.format(gb_clf.score(X_train, y_train)))
    # print('Accuracy Score (Testing): {0:.3f}'.format(gb_clf.score(X_test, y_test)))

l_rate = lr_list[np.argmax(testing)]

gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=l_rate, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_test)

print(f'Accuracy Score (GradientBoosting with {l_rate} learning rate):', accuracy_score(y_test, predictions))
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('Classification Report:')
print(classification_report(y_test, predictions))

"""Hyperparameter Tuning using GridSearchCV"""
print('Hyperparameter Tuning using GridSearchCV')
param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}

gb_model = GradientBoostingClassifier()

grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print('Best Parameters:', best_params)
print('Best Model:', best_model)
print('Best Score:', grid_search.best_score_)

y_predictions_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_predictions_best)
precision_best = precision_score(y_test, y_predictions_best)
recall_best = recall_score(y_test, y_predictions_best)
f1_best = f1_score(y_test, y_predictions_best)
print(f'Best Model Accuracy (GridSearchCV): {accuracy_best}')
print(f'Best Model Precision (GridSearchCV): {precision_best}')
print(f'Best Model Recall (GridSearchCV): {recall_best}')
print(f'Best Model F1 Score (GridSearchCV): {f1_best}')

"""Hyperparameter Tuning using RandomizedSearchCV"""
print('Hyperparameter Tuning using RandomizedSearchCV')
param_dist = {'n_estimators': np.arange(50, 251, 50), 'learning_rate': np.linspace(0.01, 0.2, 10),
              'max_depth': np.arange(3, 8)}

gb_model = GradientBoostingClassifier()

random_search = RandomizedSearchCV(estimator=gb_model, param_distributions=param_dist, n_iter=10, cv=5,
                                   scoring='accuracy', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params_random = random_search.best_params_
best_model_random = random_search.best_estimator_
print('Best Parameters:', best_params_random)
print('Best Model:', best_model_random)
print('Best Score:', random_search.best_score_)

y_predictions_best_random = best_model_random.predict(X_test)

accuracy_best_random = accuracy_score(y_test, y_predictions_best_random)
print(f'Best Model Accuracy (RandomizedSearchCV): {accuracy_best_random}')

"""Gradient Boosting Classifier (Diamonds Dataset)"""
print('\nGradient Boosting Classifier (Diamonds Dataset)')
diamonds = sns.load_dataset('diamonds')
print(diamonds.head())
# print(diamonds.dtypes)

X = diamonds.drop('cut', axis=1)
y = diamonds['cut']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features
categorical_features = X.select_dtypes(include=['category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# print(categorical_features)
# print(numerical_features)

# Define preprocessing steps for categorical and numerical features
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features),
    ('num', StandardScaler(), numerical_features),
])
print(preprocessor)

# Create a Gradient Boosting Classifier pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42)),
])

# Perform 5-fold cross-validation
cv_score = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)

print(f'Mean Cross-Validation Accuracy: {cv_score.mean():.4f}')
print('Classification Report:\n', report)
