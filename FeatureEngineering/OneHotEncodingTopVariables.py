import pandas as pd
import numpy as np

"""One Hot Encoding of Top Variables - variables with many categories
This procedure is effective when a dataset contains multiple categorical features and each categorical feature
contains multiple category or labels (like 10 to 20)."""
# Dataset: https://www.kaggle.com/code/aditya1702/mercedes-benz-data-exploration/data

data = pd.read_csv('../Datasets/mercedesbenz.csv', usecols=['X1', 'X2', 'X3', 'X4', 'X5', 'X6'])
print(data.head())

for col in data:
    print(col, ':', data[col].unique())
    print('Unique values in ', col, ':', len(data[col].unique()))

# Find out how many labels each variable has
for col in data.columns:
    print(col, ':', len(data[col].unique()), 'labels')

print(data.shape)

# Let's examine how many columns we will obtain after one hot encoding these variables
# We can observe that, with just 6 categorical features we are getting 117 features with the help of one hot encoding.
print(pd.get_dummies(data, drop_first=True).shape)  # 117 new variables for 6 initial categorical variables

# Solution: KDD Cup Orange Challenge with Ensemble
# The Team suggested using 10 most frequent labels convert them into dummy variables using onehotencoding

# Find the top 10 most frequent categories for the variable X2
print(data.X2.value_counts().sort_values(ascending=False).head(20))

# Make a list with the most frequent categories of the variable
top_10_labels_X2 = [x for x in data.X2.value_counts().sort_values(ascending=False).head(10).index]
print(top_10_labels_X2)

# Now make the 10 binary variables
for label in top_10_labels_X2:
    data[label] = np.where(data['X2'] == label, 1, 0)
print(data[['X2'] + top_10_labels_X2].head(20))


# get whole set of dummy variables, for all the categorical variables
def one_hot_encoding_top_x(df, variable, top_x_labels):
    # function to create the dummy variables for the most frequent variables
    # we can vary the number of most frequent labels that we encode
    for label in top_x_labels:
        df[variable + '_' + label] = np.where(df[variable] == label, 1, 0)


data = pd.read_csv('../Datasets/mercedesbenz.csv', usecols=['X1', 'X2', 'X3', 'X4', 'X5', 'X6'])

# encode X2 into the top 10 most frequent categories
one_hot_encoding_top_x(data, 'X2', top_10_labels_X2)
print(data.head())

# Find the top 10 most frequent categories for X1
top_10_labels_X1 = [x for x in data.X1.value_counts().sort_values(ascending=False).head(10).index]

# now create the 10 most frequent dummy variables for X1
one_hot_encoding_top_x(data, 'X1', top_10_labels_X1)
print(data.head())

"""
Advantages:
1. Straightforward to implement.
2. Does not require hrs of variable exploration
3. Does not expand massively the feature space (number of columns in the dataset)
Disadvantages:
1. Does not add any information that may make the variable more predictive.
2. Does not keep the information of the ignored labels.
"""
