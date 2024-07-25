import seaborn as sns
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Lazy Predict help build a lot of basic models without much code and helps understand which models works better
# without any parameter tuning (pip install lazypredict)

"""LazyClassifier"""
print('LazyClassifier')
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=123)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
print(models.columns)
print(models[['Accuracy', 'ROC AUC', 'F1 Score']])

"""Titanic Dataset"""
df = sns.load_dataset('titanic')
X = df.drop('survived', axis=1)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=123)

clf = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
print(models.columns)
print(models[['Accuracy', 'ROC AUC', 'F1 Score']])

"""LazyRegressor"""
print('\nLazyRegressor')
california = fetch_california_housing()
X, y = shuffle(california.data, california.target, random_state=13)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
print(models.columns)
print(models[['Adjusted R-Squared', 'R-SquaredC', 'RMSE']])
