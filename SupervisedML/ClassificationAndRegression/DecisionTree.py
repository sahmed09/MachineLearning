import six
import sys
import pydotplus
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import Image
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')
sys.modules['sklearn.externals.six'] = six

"""Pima Indians Diabetes Database"""
print('Pima Indians Diabetes Database')
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# pima = pd.read_csv('../../Datasets/diabetes.csv', header=None, names=col_names)

pima = pd.read_csv('../../Datasets/diabetes.csv')
print(pima.head())

# X = pima.iloc[:, :-1]
# y = pima.iloc[:, -1]
# print(X.head())
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#
# scalar = StandardScaler()
# X_train = scalar.fit_transform(X_train)
# X_test = scalar.transform(X_test)

X = pima.drop('Outcome', axis=1)
y = pima['Outcome']
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
f1 = f1_score(y_test, y_predictions)
precision = precision_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions)

print('Accuracy:', accuracy)
print('F-1 Score:', f1)
print("Precision:", precision)
print("Recall:", recall)

c_matrix = confusion_matrix(y_test, y_predictions)
print(c_matrix)

sns.heatmap(c_matrix, annot=True)
plt.show()

report = classification_report(y_test, y_predictions)
print(report)

print('Predicted (0): ', classifier.predict([[1, 150, 82, 42, 475, 40.6, 0.680, 23]]))
print('Predicted (1): ', classifier.predict([[0, 162, 76, 56, 110, 51.2, 0.749, 25]]))

tree.plot_tree(classifier, feature_names=X.columns.tolist(), filled=True)
plt.show()

# Visualizing Decision Trees
dot_data = six.StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                feature_names=X.columns.tolist(), class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Figures/diabetes.png')
Image(graph.create_png())

# Optimizing Decision Tree Performance
# criterion='entropy' is used to measure the quality of split, that is calculated by information gain given by entropy.
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3)
classifier = classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
f1 = f1_score(y_test, y_predictions)
precision = precision_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions)

print('Accuracy:', accuracy)
print('F-1 Score:', f1)
print("Precision:", precision)
print("Recall:", recall)

c_matrix = confusion_matrix(y_test, y_predictions)
print(c_matrix)

sns.heatmap(c_matrix, annot=True)
plt.show()

report = classification_report(y_test, y_predictions)
print(report)

print('Predicted (0): ', classifier.predict([[1, 150, 82, 42, 475, 40.6, 0.680, 23]]))
print('Predicted (1): ', classifier.predict([[0, 162, 76, 56, 110, 51.2, 0.749, 25]]))

tree.plot_tree(classifier, feature_names=X.columns.tolist(), filled=True)
plt.show()

# Visualizing Decision Trees
dot_data = six.StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                feature_names=X.columns.tolist(), class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Figures/diabetes_after_optimization.png')
Image(graph.create_png())

"""User Dataset"""
print('\nUser Dataset')
data_set = pd.read_csv('../../Datasets/User_Data.csv')

feature_cols = ['Age', 'EstimatedSalary']
X = data_set.iloc[:, [2, 3]].values
y = data_set.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
f1 = f1_score(y_test, y_predictions)
precision = precision_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions)

print('Accuracy(in %):', accuracy * 100)
print('F-1 Score:', f1)
print("Precision:", precision)
print("Recall:", recall)

c_matrix = confusion_matrix(y_test, y_predictions)
print(c_matrix)

sns.heatmap(c_matrix, annot=True)
plt.show()

report = classification_report(y_test, y_predictions)
print(report)

print('Predicted [1]:', classifier.predict(scaler.transform([[55, 86000]])))
print('Predicted [0]:', classifier.predict(scaler.transform([[35, 86000]])))

# Visualising the Training set results
x_set, y_set = X_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('purple', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('purple', 'green'))(i), label=j)
plt.title('Decision Tree Algorithm (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
x_set, y_set = X_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('purple', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('purple', 'green'))(i), label=j)
plt.title('Decision Tree Algorithm(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# visualize the tree
dot_data = six.StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Figures/user_data.png')
Image(graph.create_png())

# Optimizing the Decision Tree Classifier
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
f1 = f1_score(y_test, y_predictions)
precision = precision_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions)

print('Accuracy(in %):', accuracy * 100)
print('F-1 Score:', f1)
print("Precision:", precision)
print("Recall:", recall)

c_matrix = confusion_matrix(y_test, y_predictions)
print(c_matrix)

sns.heatmap(c_matrix, annot=True)
plt.show()

report = classification_report(y_test, y_predictions)
print(report)

print('Predicted [1]:', classifier.predict(scaler.transform([[55, 86000]])))
print('Predicted [0]:', classifier.predict(scaler.transform([[35, 86000]])))

# visualize the tree
dot_data = six.StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Figures/user_data_after_optimization.png')
Image(graph.create_png())

df = pd.read_csv('../../Datasets/data.csv')

df['Nationality'] = df['Nationality'].map({'UK': 0, 'USA': 1, 'N': 2})
df['Go'] = df['Go'].map({'YES': 1, 'NO': 0})
print(df)

features = ['Age', 'Experience', 'Rank', 'Nationality']
# X = df[features]
X = df.drop('Go', axis=1)
y = df['Go']
print(X.columns)
# print(X.head())
# print(y.head())

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features, filled=True)
plt.show()

print('Predict (YES):', dtree.predict([[40, 10, 7, 1]]))
print('Predict (NO):', dtree.predict([[40, 10, 6, 1]]))
