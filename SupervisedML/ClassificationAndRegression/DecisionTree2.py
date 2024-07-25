import six
import sys
import pydotplus
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from IPython.display import Image
from graphviz import Source
from sklearn import tree
from sklearn.datasets import load_iris, load_diabetes, make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')
sys.modules['sklearn.externals.six'] = six

"""IRIS Dataset -> Classification Problem"""
iris = load_iris()
X = iris.data[:, 2:]  # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
tree_clf.fit(X, y)

# plt.figure(figsize=(15, 10))
tree.plot_tree(tree_clf, filled=True, rounded=True, feature_names=iris.feature_names[2:], class_names=iris.target_names)
plt.show()

export_graphviz(tree_clf, out_file="Figures/iris_tree.dot", rounded=True, filled=True,
                feature_names=iris.feature_names[2:], class_names=iris.target_names)

with open("Figures/iris_tree.dot") as f:
    dot_graph = f.read()

Source(dot_graph)
graph = pydotplus.graph_from_dot_data(dot_graph)
graph.write_png('Figures/iris_tree.png')
Image(graph.create_png())

dot_data = tree.export_graphviz(tree_clf, out_file=None, filled=True, rounded=True,
                                feature_names=iris.feature_names[2:], class_names=iris.target_names)
graph = graphviz.Source(dot_data, filename='Figures/iris', format="png")
graph.view()

"""Diabetes Dataset -> Regression Problem"""
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

tree_reg = DecisionTreeRegressor(criterion='squared_error', max_depth=2)
tree_reg.fit(X, y)

dot_data = six.StringIO()
export_graphviz(tree_reg, out_file=dot_data, rounded=True, filled=True,
                feature_names=diabetes.feature_names, class_names=diabetes.target)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Figures/diabetes_tree.png')
Image(graph.create_png())

"""Synthetic Dataset"""
X, y = make_classification(100, 5, n_classes=2, shuffle=True, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)

model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)
predicted_value = model.predict(X_test)
print(predicted_value)

tree.plot_tree(model)

zeroes = 0
ones = 0
for i in range(0, len(y_train)):
    if y_train[i] == 0:
        zeroes += 1
    else:
        ones += 1

print('Zeroes:', zeroes)
print('Ones:', ones)

val = 1 - ((zeroes / 70) * (zeroes / 70) + (ones / 70) * (ones / 70))
print("Gini :", val)

match = 0
UnMatch = 0

for i in range(30):
    if predicted_value[i] == y_test[i]:
        match += 1
    else:
        UnMatch += 1

accuracy = match / 30
print("Accuracy is:", accuracy)
