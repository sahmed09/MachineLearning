import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# KNN Classifier
df = pd.read_csv('../../Datasets/Classified Data', index_col=0)
print(df.head())

# Standardize the Variables
# The KNN classifier predicts the class of a given test observation by identifying the observations that are nearest
# to it, the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on
# the distance between the observations, and hence on the KNN classifier, than variables that are on a small scale.

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat.head())

# Pair Plot
sns.pairplot(df, hue='TARGET CLASS')
plt.show()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.30)

# Using KNN
# we are trying to come up with a model to predict whether someone will TARGET CLASS or not. We'll start with k=1.
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# Predictions and Evaluations
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

"""Choosing a K Value"""
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K-value')
plt.ylabel('Error Rate')
plt.show()

# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# NOW WITH K=23
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

"""Using cross-validation"""
accuracy_rate = []
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn, df_feat, df['TARGET CLASS'], cv=10)
    accuracy_rate.append(score.mean())
    error_rate.append(1 - score.mean())
# print(accuracy_rate)
# print(error_rate)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), accuracy_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
         markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K-value')
plt.ylabel('Accuracy Rate')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K-value')
plt.ylabel('Error Rate')
plt.show()

# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
# knn = KNeighborsClassifier(n_neighbors=1)
# score = cross_val_score(knn, df_feat, df['TARGET CLASS'], cv=10)
# print(score)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# NOW WITH K=23
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
