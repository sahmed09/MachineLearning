import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

# KNN Classifier
df = pd.read_csv('../../Datasets/Classified Data', index_col=0)
print(df.head())

# Normalizing & Splitting the Data
# Split the data into features (X) and target (y)
X = df.drop('TARGET CLASS', axis=1)
y = df['TARGET CLASS']

# scaler = StandardScaler()
# scaler.fit(df.drop('TARGET CLASS', axis=1))
# scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
# X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.30)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting and Evaluating the Model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Predictions and Evaluations
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Using Cross Validation to Get the Best Value of k
k_values = [i for i in range(1, 31)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(X)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))
print(scores)

sns.lineplot(x=k_values, y=scores, marker='o')
plt.title('Accuracy Rate vs. K Value')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.show()

# More Evaluation Metrics
best_index = np.argmax(scores)
best_k = k_values[best_index]
print(best_index)
print(best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print()

"""Another Example"""
data = pd.read_csv('../../Datasets/User_Data.csv')
print(data.head())
print(data.isnull().sum())

X = data.iloc[:, [2, 3]].values  # independent variables are age and salary at index 2 and 3
y = data.iloc[:, 4].values  # dependent variable purchased is at index 4

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.25, random_state=0)

scalar = StandardScaler()
X_Train = scalar.fit_transform(X_Train)
X_Test = scalar.transform(X_Test)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)  # metric='euclidean'
classifier.fit(X_Train, Y_Train)
y_predictions = classifier.predict(X_Test)
print(y_predictions)

accuracy = accuracy_score(Y_Test, y_predictions)
print("Accuracy:", accuracy)

print(confusion_matrix(Y_Test, y_predictions))
print(classification_report(Y_Test, y_predictions))

print('Predicted [1]:', classifier.predict(scalar.transform([[55, 86000]])))
print('Predicted [0]:', classifier.predict(scalar.transform([[35, 86000]])))

x_set, y_set = X_Train, Y_Train
# x_set, y_set = X_Test, Y_Test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('K-NN Algorithm (Training set)')
# plt.title('K-NN Algorithm (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
