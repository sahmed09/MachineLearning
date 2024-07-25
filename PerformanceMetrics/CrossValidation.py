import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, LeaveOneOut, LeavePOut
from sklearn.model_selection import train_test_split, ShuffleSplit, TimeSeriesSplit

cancer = pd.read_csv('../Datasets/cancer_dataset.csv')
print(cancer.head())

# Independent And dependent features
X = cancer.iloc[:, 2:]
y = cancer.iloc[:, 1]
print(X.shape, y.shape)

X = X.dropna(axis=1)
print(X.shape, y.shape)
print(y.value_counts())

"""Approach-1: HoldOut Validation Approach - Train And Test Split"""
print('\nApproach-1: Train And Test Split')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print('Train And Test Split Score:', result)

"""Approach-2: K Fold Cross Validation"""
print('\nApproach-2: K Fold Cross Validation')
model = DecisionTreeClassifier()
kfold_validation = KFold(n_splits=10)
results = cross_val_score(model, X, y, cv=kfold_validation)
print(results)
print('Min Score:', min(results))
print('Max Score:', max(results))
print('K Fold Cross Validation Score:', np.mean(results))

"""Approach-3: Stratified K-fold Cross Validation (Works good with imbalanced dataset)"""
print('\nApproach-3: Stratified K-fold Cross Validation')
skfold = StratifiedKFold(n_splits=5)
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=skfold)
print(scores)
print('Min Score:', min(scores))
print('Max Score:', max(scores))
print('Stratified K-fold Cross Validation:', np.mean(scores))

"""Approach-4: Leave One Out Cross Validation (LOOCV)"""
print('\nApproach-4: Leave One Out Cross Validation(LOOCV)')
model = DecisionTreeClassifier()
leave_validation = LeaveOneOut()
results = cross_val_score(model, X, y, cv=leave_validation)
print(results)
print('Leave One Out Cross Validation (LOOCV) Score:', np.mean(results))

"""Approach-5: Leave-p-out cross-validation"""
print('\nApproach-5: Leave-p-out cross-validation')
model = DecisionTreeClassifier()
lpo = LeavePOut(p=1)
lpo.get_n_splits(X)
results = cross_val_score(model, X, y, cv=lpo)
print(results)
print('Leave-p-out Cross Validation Score:', np.mean(results))

"""Approach-6: Repeated Random Test-Train Splits
This technique is a hybrid of traditional train-test splitting and the k-fold cross-validation method. 
In this technique, we create random splits of the data in the training-test set manner and then repeat the 
process of splitting and evaluating the algorithm multiple times, just like the cross-validation method."""
print('\nApproach-6: Repeated Random Test-Train Splits')
model = DecisionTreeClassifier()
ssplit = ShuffleSplit(n_splits=10, test_size=0.3)
results = cross_val_score(model, X, y, cv=ssplit)
print(results)
print('Repeated Random Test-Train Splits Score:', np.mean(results))

"""Approach-7: Time series cross-validation"""
print('\nApproach-7: Time series cross-validation')
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 1], [2, 2]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
tscv = TimeSeriesSplit(n_splits=7)
print(tscv)
i = 1
for train_index, test_index in tscv.split(X):
    print(f'Train {i}:', train_index, 'Test:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    i += 1
