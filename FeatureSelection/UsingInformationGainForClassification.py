import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest

"""Feature Selection-Information gain - mutual information In Classification Problem Statements
Mutual Information (MI)
MI Estimate mutual information for a discrete target variable.
MI between two random variables is a non-negative value, which measures the dependency between the variables. It is
equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
The function relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances.

A quantity called mutual information measures the amount of information one can obtain from one random variable give
another. The mutual information between two random variables X and Y can be stated formally as follows:
I(X ; Y) = H(X) – H(X | Y) Where I(X ; Y) is the mutual information for X and Y, H(X) is the entropy for X and
H(X | Y) is the conditional entropy for X given Y. The result has the units of bits."""

df = pd.read_csv(
    'https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv')
print(df.head())
print(df['Wine'].unique())
print(df.info())

X = df.drop(labels=['Wine'], axis=1)
y = df['Wine']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.head())

# determine the mutual information
mutual_info = mutual_info_classif(X_train, y_train)
print(mutual_info)

mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
print(mutual_info.sort_values(ascending=False))

# plot the ordered mutual_info values per feature
mutual_info.sort_values(ascending=False).plot.bar(figsize=(16, 7))
plt.show()

# Now we Will select the top 5 important features
top_five_features = SelectKBest(mutual_info_classif, k=5)
top_five_features.fit(X_train, y_train)

features = X_train.columns[top_five_features.get_support()]
print(features)

"""Difference Between Information Gain And Mutual Information
I(X ; Y) = H(X) – H(X | Y) and IG(S, a) = H(S) – H(S | a)
As such, mutual information is sometimes used as a synonym for information gain. Technically, they calculate the 
same quantity if applied to the same data.

Comparion of Univariate And Mutual Information
Comparison of F-test and mutual information 
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py"""
