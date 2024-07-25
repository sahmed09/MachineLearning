import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

"""Naive Bayes Classifier with USER Dataset"""
dataset = pd.read_csv('../../Datasets/User_data.csv')

X = dataset.iloc[:, [2, 3]].values  # independent variables are age and salary at index 2 and 3
y = dataset.iloc[:, 4].values  # dependent variable purchased is at index 4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

classifier = GaussianNB()
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

print('Predicted [1]:', classifier.predict(scalar.transform([[55, 86000]])))
print('Predicted [0]:', classifier.predict(scalar.transform([[35, 86000]])))

# Visualising the Training set results
x_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('purple', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('purple', 'green'))(i), label=j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
x_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('purple', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('purple', 'green'))(i), label=j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

"""Naive Bayes Classifier with INCOME Evaluation Dataset"""
# https://www.kaggle.com/code/shaistashaikh/income-classification

income_data = pd.read_csv('../../Datasets/income_evaluation.csv')
income_data.columns = income_data.columns.str.lstrip()  # Cleaning left spaces of column name

print(income_data.head())
print(income_data.shape)
print(income_data['sex'].head())
print(income_data.describe())
print(income_data.info())
print(income_data.isnull().sum())  # Test null values
print(income_data.isna().sum())  # Test NA values
print(income_data.columns)

# Fill Missing Category Entries
# income_data["workclass"] = income_data["workclass"].fillna("X")
# income_data["occupation"] = income_data["occupation"].fillna("X")
# income_data["native-country"] = income_data["native-country"].fillna("United-States")

gender = {" Male": 1, " Female": 2}
income_data['sex'] = income_data['sex'].map(gender)
print(income_data['sex'].head())
print(income_data.info())

numeric_features = income_data.columns[income_data.dtypes != 'O']
categorical_feature = income_data.columns[income_data.dtypes == 'O']
print('Numerical Features:', numeric_features)
print('Categorical Features:', categorical_feature)

# check the distribution of target variable, 'income'
target_ratio = income_data['income'].value_counts() / len(income_data)
print(target_ratio)
print('Target Ratio Index:', target_ratio.index)

# Check for data imbalance
plt.figure(figsize=(6, 6))
plt.bar(target_ratio.index, target_ratio)
plt.ylabel('Percentage')
plt.show()

sns.countplot(income_data['income'], label="Count")
plt.title('Income Data')
plt.show()

sns.heatmap(income_data[numeric_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

sns.FacetGrid(income_data, col='income').map(sns.distplot, "age")
plt.show()

# encode target variable 'income'
label_encoder = LabelEncoder()
income_data.income = label_encoder.fit_transform(income_data.income)
print(label_encoder.classes_)
# print(income_data['income'])
# print(income_data.head())

numerical_features = income_data.columns[income_data.dtypes != 'O']
print(*numerical_features, sep=' | ')

# convert the categorical features to dummy data
categorical_features = income_data.columns[income_data.dtypes == 'O']
print(*categorical_features, sep=' | ')

new_data = pd.get_dummies(income_data, columns=categorical_features)
print(new_data.head())

target = list(income_data.columns)[14]
print('Target Class:', target)

# split data into train and test data
X = new_data
y = new_data.income
# x = income_data.drop(['income'], axis=1)
# y = income_data[['income']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# GaussianNB
print('GaussianNB')
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)
gaussian_nb_predictions = gaussian_nb.predict(X_test)

acc_gnb = accuracy_score(y_test, gaussian_nb_predictions)
f1_gnb = f1_score(y_test, gaussian_nb_predictions)
precision_gnb = precision_score(y_test, gaussian_nb_predictions)
recall_gnb = recall_score(y_test, gaussian_nb_predictions)

print('Accuracy Score (GaussianNB): ', acc_gnb)
print('F-1 Score (GaussianNB):', f1_gnb)
print("Precision (GaussianNB):", precision_gnb)
print("Recall (GaussianNB):", recall_gnb)

c_matrix = confusion_matrix(y_test, gaussian_nb_predictions)
print(c_matrix)

report = classification_report(y_test, gaussian_nb_predictions)
print(report)

# BernoulliNB
print('BernoulliNB')
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train, y_train)
bernoulli_nb_predictions = bernoulli_nb.predict(X_test)

acc_bnb = accuracy_score(y_test, bernoulli_nb_predictions)
f1_bnb = f1_score(y_test, bernoulli_nb_predictions)
precision_bnb = precision_score(y_test, bernoulli_nb_predictions)
recall_bnb = recall_score(y_test, bernoulli_nb_predictions)

print('Accuracy Score (BernoulliNB): ', acc_bnb)
print('F-1 Score (BernoulliNB):', f1_bnb)
print("Precision (BernoulliNB):", precision_bnb)
print("Recall (BernoulliNB):", recall_bnb)

c_matrix = confusion_matrix(y_test, bernoulli_nb_predictions)
print(c_matrix)

report = classification_report(y_test, bernoulli_nb_predictions)
print(report)

plt.figure(figsize=(6, 6))
plt.bar(['GaussianNB', 'BernoulliNB'], [acc_gnb, acc_bnb])
plt.ylabel('Accuracy Score')
plt.show()
