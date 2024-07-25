import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                col_name = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(col_name)
    return col_corr


data = pd.read_csv('../Datasets/BostonHousing.csv')
print(data.head())

X = data.drop('medv', axis=1)
y = data['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
print(X_train.corr())

# Using Pearson Correlation
plt.figure(figsize=(12, 10))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

corr_features = correlation(X_train, 0.7)
print(len(corr_features))
print(corr_features)

X_train.drop(corr_features, axis=1, inplace=True)
X_test.drop(corr_features, axis=1, inplace=True)
print(X_train.shape, X_test.shape)

print('\nSantander_Customer_Satisfaction')
santander = pd.read_csv('../Datasets/Santander_Customer_Satisfaction.csv', nrows=10000)
X = santander.drop(labels=['TARGET'], axis=1)
y = santander['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, X_test.shape)

# Using Pearson Correlation
corrmat = X_train.corr()
fig, ax = plt.subplots()
fig.set_size_inches(11, 11)
sns.heatmap(corrmat)
plt.show()

corr_features = correlation(X_train, 0.9)
print(len(set(corr_features)))
print(corr_features)

X_train = X_train.drop(corr_features, axis=1)
X_test = X_test.drop(corr_features, axis=1)
print(X_train.shape, X_test.shape)
