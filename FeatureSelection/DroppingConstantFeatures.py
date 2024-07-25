import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# Feature Selection- Dropping Constant Features Using Variance Threshold
# Variance Threshold
# Feature selector that removes all low-variance features.
# This feature selection algorithm looks only at the features (X), not the desired outputs (y), and can thus be
# used for unsupervised learning.
# It will remove zero variance features.

data = pd.DataFrame({"A": [1, 2, 4, 1, 2, 4],
                     "B": [4, 5, 6, 7, 8, 9],
                     "C": [0, 0, 0, 0, 0, 0],
                     "D": [1, 1, 1, 1, 1, 1]})
print(data)

var_thres = VarianceThreshold(threshold=0)
var_thres.fit(data)

print(var_thres.get_support())
print(data.columns[var_thres.get_support()])

constant_columns = [column for column in data.columns if column not in data.columns[var_thres.get_support()]]
print('Constant Columns:', constant_columns)
print(len(constant_columns))

for feature in constant_columns:
    print(feature)

data.drop(constant_columns, axis=1, inplace=True)
print(data)

df = pd.read_csv('../Datasets/Santander_Customer_Satisfaction.csv', nrows=10000)
print(df.shape)
print(df.head())

X = df.drop(labels=['TARGET'], axis=1)
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, X_test.shape)

var_thres = VarianceThreshold(threshold=0)
var_thres.fit(X_train)

print(var_thres.get_support())
print(sum(var_thres.get_support()))

# Find non-constant features
print(print(len(X_train.columns[var_thres.get_support()])))

constant_columns = [column for column in X_train.columns if column not in X_train.columns[var_thres.get_support()]]
print('Constant Columns:', len(constant_columns))
print(constant_columns)

X_train = X_train.drop(constant_columns, axis=1)
print(X_train.head())
