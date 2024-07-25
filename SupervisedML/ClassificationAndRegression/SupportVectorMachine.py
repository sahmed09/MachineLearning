import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.inspection import DecisionBoundaryDisplay
import warnings

warnings.filterwarnings('ignore')

"""IRIS Dataset"""
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_predictions = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
print('Accuracy:', accuracy)

# define the parameter grid
parameter_grid = {'C': [0.1, 1, 10, 100],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'degree': [2, 3, 4],
                  'coef0': [0.0, 0.1, 0.5],
                  'gamma': ['scale', 'auto']}

svm = SVC()

# perform grid search to find the best set of parameters
grid_search = GridSearchCV(svm, parameter_grid, cv=5)
grid_search.fit(X_train, y_train)

print('Best parameters:', grid_search.best_params_)
print('Best accuracy:', grid_search.best_score_)

"""Breast Cancer Dataset"""
cancer = load_breast_cancer()

print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)
print(cancer.data.shape)
print(cancer.data[0:5])
print(cancer.target)  # print the cancer labels (0:malignant, 1:benign)

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
precision = precision_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

X = cancer.data[:, :2]
y = cancer.target
svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
svm.fit(X, y)
# Polynomial and RBF are useful for non-linear hyperplane.
# C is the penalty parameter (Regularization), which represents misclassification or error term.
# A lower value of Gamma will loosely fit the training dataset
# (considers only nearby points in calculating the separation line)

DecisionBoundaryDisplay.from_estimator(svm, X, response_method="predict", cmap=plt.cm.Spectral, alpha=0.8,
                                       xlabel=cancer.feature_names[0], ylabel=cancer.feature_names[1], )
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors="k")
plt.show()

"""User Dataset"""
data_set = pd.read_csv('../../Datasets/User_Data.csv')

x = data_set.iloc[:, [2, 3]].values
y = data_set.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
f1 = f1_score(y_test, y_predictions)
precision = precision_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions)

print('Gaussian Naive Bayes model accuracy(in %):', accuracy * 100)
print('F-1 Score:', f1)
print("Precision:", precision)
print("Recall:", recall)

c_matrix = confusion_matrix(y_test, y_predictions)
print(c_matrix)

sns.heatmap(c_matrix, annot=True)
plt.show()

report = classification_report(y_test, y_predictions)
print(report)

print('Predicted [1]:', classifier.predict(scaler.transform([[55, 86000]])))
print('Predicted [0]:', classifier.predict(scaler.transform([[35, 86000]])))

# Visualising the Training set results
x_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
x_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
