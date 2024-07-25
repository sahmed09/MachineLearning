import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import EasyEnsembleClassifier
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('../Datasets/creditcard_fraud.csv')  # Class 0 = Normal Transaction, Class 1 = Fraud Transaction
print(data.shape)
print(data.head())
print(data.info())
print(data['Class'].value_counts())

# Independent and Dependent Features
X = data.drop('Class', axis=1)
y = data.Class

# Cross Validation Like KFOLD and Hyperparameter Tuning
print('\nCross Validation Like KFOLD and Hyperparameter Tuning')
print('Logistic Regression')

log_class = LogisticRegression()
grid = {'C': 10.0 ** np.arange(-2, 3), 'penalty': ['l1', 'l2']}
cv = KFold(n_splits=5, random_state=None, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
clf = GridSearchCV(log_class, grid, cv=cv, n_jobs=-1, scoring='f1_macro')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy Score (LogisticRegression):', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(y_train.value_counts())

print('\nRandom Forest Classifier')
class_weight = dict({0: 1, 1: 100})

classifier = RandomForestClassifier(class_weight=class_weight)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy Score (RandomForestClassifier):', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Under Sampling (Not wise to use - only use in case of small dataset)
print('\nUnder Sampling')
print(Counter(y_train))

ns = NearMiss()
X_train_ns, y_train_ns = ns.fit_resample(X_train, y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))

classifier = RandomForestClassifier()
classifier.fit(X_train_ns, y_train_ns)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy Score (RandomForestClassifier-Under Sampling):', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Over Sampling
print('\nOver Sampling - RandomOverSampler')

os = RandomOverSampler()
X_train_ns, y_train_ns = os.fit_resample(X_train, y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))

classifier = RandomForestClassifier()
classifier.fit(X_train_ns, y_train_ns)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy Score (RandomForestClassifier-over Sampling):', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SMOTETomek
print('\nOver Sampling - SMOTETomek')

os = SMOTETomek()
X_train_ns, y_train_ns = os.fit_resample(X_train, y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))

classifier = RandomForestClassifier()
classifier.fit(X_train_ns, y_train_ns)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy Score (RandomForestClassifier-SMOTETomek):', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Ensemble Techniques
print('\nOver Sampling - Ensemble Techniques')

easy = EasyEnsembleClassifier()
easy.fit(X_train, y_train)

y_pred = easy.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy Score (RandomForestClassifier-Ensemble Techniques):', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
