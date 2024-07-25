import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

X, y = load_breast_cancer(return_X_y=True)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_prediction = logistic_regression.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction, average='weighted')
recall = recall_score(y_test, y_prediction, average='weighted')
f1 = f1_score(y_test, y_prediction, average='weighted')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

y_probs = logistic_regression.predict_proba(X_test)
auc_score = roc_auc_score(y_test, y_probs[:, 1])
print('AUC Score:', auc_score)

c_matrix = confusion_matrix(y_test, y_prediction)
print(c_matrix)

report = classification_report(y_test, y_prediction)
print(report)

fpr, tpr, threshold = roc_curve(y_test, y_prediction)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0.2, 0.8, 0.6, 0.4, 0.9, 0.1]

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
