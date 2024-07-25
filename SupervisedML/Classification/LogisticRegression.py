import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

"""Binomial Logistic Regression"""
print('Binomial Logistic Regression')

data = pd.read_csv('../../Datasets/User_Data.csv')
print(data.head())
print(data.isnull().sum())

X = data.iloc[:, [2, 3]].values  # independent variables are age and salary at index 2 and 3
y = data.iloc[:, 4].values  # dependent variable purchased is at index 4

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.25, random_state=0)

scalar = StandardScaler()
X_Train = scalar.fit_transform(X_Train)
X_Test = scalar.transform(X_Test)

classifier = LogisticRegression(random_state=0)  # we get the same train and test sets across different executions
classifier.fit(X_Train, Y_Train)

y_predictions = classifier.predict(X_Test)
print(y_predictions)

accuracy = accuracy_score(Y_Test, y_predictions)
print('Accuracy:', accuracy)

c_matrix = confusion_matrix(Y_Test, y_predictions)
print(c_matrix)

sns.heatmap(pd.DataFrame(c_matrix), annot=True)
plt.show()

report = classification_report(Y_Test, y_predictions)
print(report)

print('Predicted [1]:', classifier.predict(scalar.transform([[55, 86000]])))
print('Predicted [0]:', classifier.predict(scalar.transform([[35, 86000]])))
print()

"""Binomial Logistic Regression"""
print('Binomial Logistic Regression')

X, y = load_breast_cancer(return_X_y=True)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

y_prediction = logistic_regression.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)
print('Logistic Regression model accuracy (in %):', accuracy * 100)

c_matrix = confusion_matrix(y_test, y_prediction)
print(c_matrix)

report = classification_report(y_test, y_prediction)
print(report)
print()

"""Multinomial Logistic Regression"""
print('Multinomial Logistic Regression')

digits = load_digits()
X = digits.data
y = digits.target
print(X.shape, y.shape)

# Displaying some of the images and their labels
plt.figure(figsize=(15, 4))
for index, (image, label) in enumerate(zip(X[0: 5], y[0: 5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

y_prediction = logistic_regression.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)
print('Logistic Regression model accuracy (in %):', accuracy * 100)

c_matrix = confusion_matrix(y_test, y_prediction)
print(c_matrix)

report = classification_report(y_test, y_prediction)
print(report)
print()

# Wrong predictions and actual output
index = 0
misclassifiedindex = []
for predict, actual in zip(y_prediction, y_test):
    if predict != actual:
        misclassifiedindex.append(index)
    index += 1

plt.figure(figsize=(15, 3))
for plot_index, wrong in enumerate(misclassifiedindex[0: 4]):
    plt.subplot(1, 4, plot_index + 1)
    plt.imshow(np.reshape(X_test[wrong], (8, 8)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(y_prediction[wrong], y_test[wrong]), fontsize=20)
plt.show()

"""One vs Rest"""
print('One vs Rest')

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)

# model = LogisticRegression()
# model = OneVsRestClassifier(model)
# model.fit(X, y)
model = LogisticRegression(multi_class='ovr')  # 'multi_class' parameter: {'multinomial', 'ovr', 'auto'}
model.fit(X, y)
y_pred = model.predict(X)

# print(y_pred)
print('Accuracy:', accuracy_score(y, y_pred))
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
