import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, make_classification, load_wine, make_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.model_selection import cross_val_predict, learning_curve
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

"""AdaBoost Classifier (Breast Cancer Dataset)"""
print('AdaBoost Classifier (Breast Cancer Dataset)')
breast_cancer = load_breast_cancer()

X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
print(X.head())

encoder = LabelEncoder()
binary_encoded_y = pd.Series(encoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, binary_encoded_y, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
print('Accuracy:', accuracy)

c_matrix = confusion_matrix(y_test, y_predictions)
print(c_matrix)

# Another Approach
X = breast_cancer.data
y = breast_cancer.target

print('Name of the Features:', breast_cancer.feature_names)
print('Name of the Classes:', breast_cancer.target_names)
print('Shape of the Dataset:', breast_cancer.data.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)

log_reg.fit(X_train, y_train)
adaboost.fit(X_train, y_train)

y_pred_1 = adaboost.predict(X_test)
y_pred_2 = log_reg.predict(X_test)

accuracy_log = accuracy_score(y_test, y_pred_2)
accuracy_ada = accuracy_score(y_test, y_pred_1)
print("Accuracy of Logistic Regression:", accuracy_log)
print("Accuracy of AdaBoost:", accuracy_ada)

"""AdaBoost Classifier (Digits Dataset)"""
print('\nAdaBoost Classifier (Digits Dataset)')
digit_data = load_digits()
X = digit_data['data']
y = digit_data['target']

plt.imshow(X[4].reshape(8, 8))
plt.show()

ada_boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
scores_ada = cross_val_score(ada_boost, X, y, cv=6)
print(scores_ada)
print('Mean Accuracy Score:', scores_ada.mean())

score = []
for depth in [1, 2, 10]:
    ada_boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth))
    scores_ada = cross_val_score(ada_boost, X, y, cv=6)
    score.append(scores_ada.mean())
print('Mean Accuracy Score [1, 2, 10]:', score)

"""AdaBoost Classifier (IRIS Dataset)"""
print('\nAdaBoost Classifier (IRIS Dataset)')
iris_data = sns.load_dataset('iris')
print(iris_data.shape)
print(iris_data.head())

X = iris_data.iloc[:, :-1]
y = iris_data.iloc[:, -1]
print("Shape of X is %s and shape of y is %s" % (X.shape, y.shape))

total_classes = y.nunique()
print("Number of unique species in dataset are: ", total_classes)

distribution = y.value_counts()
print(distribution)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=28)

adb = AdaBoostClassifier()
adb_model = adb.fit(X_train, y_train)

print("The accuracy of the model on validation set is", adb_model.score(X_test, y_test))

# Another Approach
iris_data = sns.load_dataset('iris')
print(iris_data.head())
print(iris_data.info())

X = iris_data.iloc[:, :-1]
y = iris_data['species']
print(X.head())
print(y.head())

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

ab_classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
model1 = ab_classifier.fit(X_train, y_train)
y_pred = model1.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('AdaBoost Classifier Model Accuracy:', accuracy)

# Further evaluation with SVC base estimator
svc = SVC(probability=True, kernel='linear')
ab_classifier = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=1, random_state=0)
model2 = ab_classifier.fit(X_train, y_train)
y_pred = model2.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy with SVC Base Estimator:', accuracy)

"""AdaBoost Classifier (Wine Dataset)"""
print('\nAdaBoost Classifier (Wine Dataset)')
wine_data = load_wine()

print('Name of the Features:', wine_data.feature_names)
print('Name of the Classes:', wine_data.target_names)
print('Shape of the Dataset:', wine_data.data.shape)

X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
y = pd.Series(wine_data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Build a Decision Tree Stump Model and get its Accuracy Score
dt_clf = DecisionTreeClassifier(max_depth=1, criterion='gini', random_state=23)
dt_clf.fit(X_train, y_train)
dt_clf_predict = dt_clf.predict(X_test)

dt_clf_accuracy = round(accuracy_score(y_test, dt_clf_predict), 3)
print(f"Decision Tree Classifier Test Accuracy Score: ", dt_clf_accuracy)

# Build an AdaBoost Classifier and get its Accuracy Score
ada_boost_clf = AdaBoostClassifier(base_estimator=dt_clf, n_estimators=50, learning_rate=0.5, random_state=23)
ada_boost_clf.fit(X_train, y_train)
ada_boost_clf_predict = ada_boost_clf.predict(X_test)

ada_boost_clf_accuracy = round(accuracy_score(y_test, ada_boost_clf_predict), 3)
print(f"Decision Tree AdaBoost Model Test Accuracy Score: ", ada_boost_clf_accuracy)

"""AdaBoost Classifier (Synthetic Dataset)"""
print('\nAdaBoost Classifier (Synthetic Dataset)')
X, y = make_classification(n_samples=2000, n_features=30, n_informative=25, n_redundant=5)
clf = AdaBoostClassifier()
clf.fit(X, y)

test_row_data = [
    [-2.56789, 1.9012436, 0.0490456, -0.945678, -3.545673, 1.945555, -7.746789, -2.4566667, -1.845677896, -1.6778994,
     2.336788043, -4.305666617, 0.466641, -1.2866634, -10.6777077, -0.766663, -3.5556621, 2.045456, 0.055673,
     0.94545456, 0.5677, -1.4567, 4.3333, 3.89898, 1.56565, -0.56565, -0.45454, 4.33535, 6.34343, -4.42424]]

y_pred = clf.predict(test_row_data)
print('Class predicted  %d' % y_pred[0])

"""AdaBoost Regressor (Synthetic Dataset)"""
print('\nAdaBoost Regressor (Synthetic Dataset)')
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=6)

model = AdaBoostRegressor()

# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
