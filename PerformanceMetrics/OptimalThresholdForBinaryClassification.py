import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# Select the Right Threshold values using ROC Curve

X, y = make_classification(n_samples=2000, n_classes=2, weights=[1, 1], random_state=1)
print(X.shape)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Random Forests Classifier
print('Random Forests Classifier')
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_train_pred = rf_model.predict_proba(X_train)
print(y_train_pred)
print('RF train roc-auc: {}'.format(roc_auc_score(y_train, y_train_pred[:, 1])))
y_test_pred = rf_model.predict_proba(X_test)
print('RF test roc-auc: {}'.format(roc_auc_score(y_test, y_test_pred[:, 1])))

# Logistic Regression
print('\nLogistic Regression')
log_classifier = LogisticRegression()
log_classifier.fit(X_train, y_train)
y_train_pred = log_classifier.predict_proba(X_train)
print('Logistic train roc-auc: {}'.format(roc_auc_score(y_train, y_train_pred[:, 1])))
y_test_pred = log_classifier.predict_proba(X_test)
print('Logistic test roc-auc: {}'.format(roc_auc_score(y_test, y_test_pred[:, 1])))

# Adaboost Classifier
print('\nAdaboost Classifier')
ada_classifier = AdaBoostClassifier()
ada_classifier.fit(X_train, y_train)
y_train_pred = ada_classifier.predict_proba(X_train)
print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, y_train_pred[:, 1])))
y_test_pred = ada_classifier.predict_proba(X_test)
print('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, y_test_pred[:, 1])))

# KNN Classifier
print('\nKNN Classifier')
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_train_pred = knn_classifier.predict_proba(X_train)
print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, y_train_pred[:, 1])))
y_test_pred = knn_classifier.predict_proba(X_test)
print('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, y_test_pred[:, 1])))

# Now focus on selecting the best threshold for maximum accuracy
pred = []
for model in [rf_model, log_classifier, ada_classifier, knn_classifier]:
    pred.append(pd.Series(model.predict_proba(X_test)[:, 1]))
final_prediction = pd.concat(pred, axis=1).mean(axis=1)
print(pd.concat(pred, axis=1))
print(final_prediction)
print('Ensemble test roc-auc: {}'.format(roc_auc_score(y_test, final_prediction)))

# Calculate the ROc Curve
fpr, tpr, thresholds = roc_curve(y_test, final_prediction)
print(thresholds)

accuracy_ls = []
for thres in thresholds:
    y_pred = np.where(final_prediction > thres, 1, 0)
    accuracy_ls.append(accuracy_score(y_test, y_pred, normalize=True))
accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)], axis=1)
accuracy_ls.columns = ['thresholds', 'accuracy']
accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
print(accuracy_ls.head())

print(accuracy_ls)


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


plot_roc_curve(fpr, tpr)
