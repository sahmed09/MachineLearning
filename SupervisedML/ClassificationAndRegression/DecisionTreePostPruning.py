import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

print(sklearn.__version__)
print(__doc__)

"""Post pruning decision trees with cost complexity pruning
The :class:DecisionTreeClassifier provides parameters such as "min_samples_leaf" and "max_depth" to prevent a tree 
from overfiting. "Cost complexity pruning" provides another option to control the size of a tree. In 
:class:DecisionTreeClassifier, this pruning technique is parameterized by the cost complexity parameter, ccp_alpha. 
Greater values of ccp_alpha increase the number of nodes pruned. Here we only show the effect of ccp_alpha on 
regularizing the trees and how to choose a ccp_alpha based on validation scores."""

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print('Accuracy:', accuracy)

plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True)
plt.show()

path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print('Number of nodes in the last tree is: {} with ccp_alpha: {}'.format(clfs[-1].tree_.node_count, ccp_alphas[-1]))
# For the remainder of this example, we remove the last element in clfs and ccp_alphas, because it is the trivial tree
# with only one node. Here we show that the number of nodes and tree depth decreases as alpha increases.

# Accuracy vs alpha for training and testing sets
# When ccp_alpha is set to zero and keeping the other default parameters of :class:DecisionTreeClassifier, the tree
# overfits, leading to a 100% training accuracy and 88% testing accuracy. As alpha increases, more of the tree is
# pruned, thus creating a decision tree that generalizes better.
# In this example, setting ccp_alpha=0.015 maximizes the testing accuracy.

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha for training and testing sets')
ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
ax.legend()
plt.show()

clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.012)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print('Accuracy:', accuracy)

plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True)
plt.show()
