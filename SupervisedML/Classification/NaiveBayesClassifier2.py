import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

"""Naive Bayes Classifier with IRIS Dataset"""
iris = load_iris()
X = iris.data
y = iris.target
# print(X)
# print(y)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_predictions = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
f1 = f1_score(y_test, y_predictions, average="weighted")
precision = precision_score(y_test, y_predictions, average='weighted')
recall = recall_score(y_test, y_predictions, average='weighted')

print('Gaussian Naive Bayes model accuracy(in %):', accuracy * 100)
print('F-1 Score:', f1)
print("Precision:", precision)
print("Recall:", recall)

c_matrix = confusion_matrix(y_test, y_predictions)
print(c_matrix)

sns.heatmap(c_matrix, annot=True)
plt.title('Confusion Matrix of IRIS Dataset')
plt.show()

report = classification_report(y_test, y_predictions, target_names=iris.target_names)
print(report)

print('Predicted:', gnb.predict([[4.6, 2.5, 1.7, 0.5]]))
print('Predicted:', gnb.predict([[6.5, 1.5, 3., 1.5]]))
print('Predicted:', gnb.predict([[6.5, 2.5, 5.7, 2.5]]))

"""Naive Bayes Classifier with Synthetic Dataset"""
X, y = make_classification(n_features=6, n_classes=3, n_samples=800, n_informative=2, random_state=1,
                           n_clusters_per_class=1)

plt.scatter(X[:, 0], X[:, 1], c=y, marker='*')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)

# Build a Gaussian Classifier
model = GaussianNB()
model.fit(X_train, y_train)
y_predictions = model.predict(X_test)

predicted = model.predict([X_test[6]])
print("Actual Value:", y_test[6])
print("Predicted Value:", predicted[0])

accuracy = accuracy_score(y_test, y_predictions)
f1 = f1_score(y_test, y_predictions, average="weighted")
precision = precision_score(y_test, y_predictions, average='weighted')
recall = recall_score(y_test, y_predictions, average='weighted')

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

c_matrix = confusion_matrix(y_test, y_predictions)
plt.title('Confusion Matrix of Synthetic Dataset')
print(c_matrix)

sns.heatmap(c_matrix, annot=True)
plt.show()

report = classification_report(y_test, y_predictions)
print(report)

"""Naive Bayes Classifier with Loan Dataset"""
df = pd.read_csv('../../Datasets/loan_data.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df['purpose'].value_counts())

sns.countplot(data=df, x='purpose', hue='not.fully.paid')
plt.xticks(rotation=45, ha='right')
plt.show()

# convert the ‘purpose’ column from categorical to integer using pandas `get_dummies` function
pre_df = pd.get_dummies(df, columns=['purpose'], drop_first=True)
print(pre_df.head())

unique_classes = pre_df['not.fully.paid'].unique()
print(unique_classes)

X = pre_df.drop('not.fully.paid', axis=1)
y = pre_df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)
print(len(X_train), len(X_test))

model_gaussian = GaussianNB()
model_gaussian.fit(X_train, y_train)
y_predictions = model_gaussian.predict(X_test)

accuracy_gaussian = accuracy_score(y_test, y_predictions)
f1_gaussian = f1_score(y_test, y_predictions)
precision_gaussian = precision_score(y_test, y_predictions)
recall_gaussian = recall_score(y_test, y_predictions)

print("Accuracy:", accuracy_gaussian)
print("F1 Score:", f1_gaussian)
print("Precision:", precision_gaussian)
print("Recall:", recall_gaussian)

c_matrix_gaussian = confusion_matrix(y_test, y_predictions)
print(c_matrix_gaussian)

sns.heatmap(c_matrix_gaussian, annot=True)
plt.title('Confusion Matrix of Loan Dataset (GaussianNB)')
plt.show()

report_gaussian = classification_report(y_test, y_predictions)
print(report_gaussian)

model_bernoulli = BernoulliNB()
model_bernoulli.fit(X_train, y_train)
y_predictions = model_bernoulli.predict(X_test)

accuracy_bernoulli = accuracy_score(y_test, y_predictions)
f1_bernoulli = f1_score(y_test, y_predictions)
precision_bernoulli = precision_score(y_test, y_predictions)
recall_bernoulli = recall_score(y_test, y_predictions)

print("Accuracy:", accuracy_bernoulli)
print("F1 Score:", f1_bernoulli)
print("Precision:", precision_bernoulli)
print("Recall:", recall_bernoulli)

c_matrix_bernoulli = confusion_matrix(y_test, y_predictions)
print(c_matrix_bernoulli)

sns.heatmap(c_matrix_bernoulli, annot=True)
plt.title('Confusion Matrix of Loan Dataset (BernoulliNB)')
plt.show()

report_bernoulli = classification_report(y_test, y_predictions)
print(report_bernoulli)
