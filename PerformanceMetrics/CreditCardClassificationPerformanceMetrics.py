import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('../Datasets/creditcard_fraud.csv')
print(df.head())

# creating target series
target = df['Class']
print(target)

# dropping the target variable from the data set
df.drop('Class', axis=1, inplace=True)
print(df.shape)

# converting them to numpy arrays
X = np.array(df)
y = np.array(target)
print(X.shape, y.shape)

# distribution of the target variable
print(len(y[y == 1]))
print(len(y[y == 0]))

# splitting the data set into train and test (75:25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# applying SMOTE to oversample the minority class
sm = SMOTE(random_state=2)
X_sm, y_sm = sm.fit_resample(X_train, y_train)
print(X_sm.shape, y_sm.shape)
print(len(y_sm[y_sm == 1]), len(y_sm[y_sm == 0]))

# Logistic Regression
print('\nLogistic Regression')
logreg = LogisticRegression()
logreg.fit(X_sm, y_sm)
y_logreg = logreg.predict(X_test)
y_logreg_prob = logreg.predict_proba(X_test)[:, 1]

# Performance metrics evaluation
print('Confusion Matrix:\n', metrics.confusion_matrix(y_test, y_logreg))
print('Accuracy:', metrics.accuracy_score(y_test, y_logreg))
print('Precision:', metrics.precision_score(y_test, y_logreg))
print('Recall:', metrics.recall_score(y_test, y_logreg))
print('AUC:', metrics.roc_auc_score(y_test, y_logreg_prob))
auc = metrics.roc_auc_score(y_test, y_logreg_prob)

# plotting the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_logreg_prob)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], 'r-.')
plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.2])
plt.title('Receiver Operating Characteristic\nLogistic Regression')
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# K Nearest Neighbors
print('\nK Nearest Neighbors')
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_sm, y_sm)
y_knn = knn.predict(X_test)
y_knn_prob = knn.predict_proba(X_test)[:, 1]

# metrics evaluation
print('Confusion Matrix\n:', metrics.confusion_matrix(y_test, y_knn))
print('Accuracy:', metrics.accuracy_score(y_test, y_knn))
print('Precision:', metrics.precision_score(y_test, y_knn))
print('Recall:', metrics.recall_score(y_test, y_knn))
print('AUC:', metrics.roc_auc_score(y_test, y_knn_prob))

# plotting the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_knn_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.show()

# Random Forest
print('\nRandom Forest')
rf = RandomForestClassifier(random_state=3)
rf.fit(X_sm, y_sm)
y_rf = rf.predict(X_test)
y_rf_prob = rf.predict_proba(X_test)[:, 1]

# Performance metrics evaluation
print('Confusion Matrix:\n', metrics.confusion_matrix(y_test, y_rf))
print('Accuracy:', metrics.accuracy_score(y_test, y_rf))
print('Precision:', metrics.precision_score(y_test, y_rf))
print('Recall:', metrics.recall_score(y_test, y_rf))
print('AUC:', metrics.roc_auc_score(y_test, y_rf_prob))
auc = metrics.roc_auc_score(y_test, y_rf_prob)

# plotting the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_rf_prob)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], 'r-.')
plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.2])
plt.title('Receiver Operating Characteristic\nRandom Forest')
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Random Forest (criterion='entropy)'
print("\nRandom Forest (criterion='entropy')")
rf = RandomForestClassifier(criterion='entropy', random_state=3)
rf.fit(X_sm, y_sm)
y_rf = rf.predict(X_test)
y_rf_prob = rf.predict_proba(X_test)[:, 1]

# Performance metrics evaluation
print('Confusion Matrix:', metrics.confusion_matrix(y_test, y_rf))
print('Accuracy:\n', metrics.accuracy_score(y_test, y_rf))
print('Precision:\n', metrics.precision_score(y_test, y_rf))
print('Recall:\n', metrics.recall_score(y_test, y_rf))
print('AUC:\n', metrics.roc_auc_score(y_test, y_rf_prob))
auc = metrics.roc_auc_score(y_test, y_rf_prob)

# plotting the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_rf_prob)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], 'r-.')
plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.2])
plt.title('Receiver Operating Characteristic\nRandom Forest (criterion=entropy)')
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
