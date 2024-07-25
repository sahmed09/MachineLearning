import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso  # for feature selection
from sklearn.feature_selection import SelectFromModel  # for feature selection

# pd.pandas.set_option('display.max_columns', None)  # to visualise al the columns in the dataframe

dataset = pd.read_csv('../Datasets/House_Price_X_train.csv')
print(dataset.head())

# drop dependent feature from dataset
X_train = dataset.drop(['Id', 'SalePrice'], axis=1)
print(X_train.shape)

# Capture the dependent feature
y_train = dataset[['SalePrice']]
print(y_train.shape)

# Apply Feature Selection
# First, specify the Lasso Regression model, and select a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.

# Then use the selectFromModel object from sklearn, which will select the features which coefficients are non-zero.
# remember to set the seed, the random state in this function

# lasso_model = Lasso(alpha=0.005, random_state=0)
# feature_sel_model = SelectFromModel(lasso_model)

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(X_train, y_train)

print(feature_sel_model.get_support())  # True means important feature, False means the feature can be skipped

# let's print the number of total and selected features
# this is how we can make a list of the selected features
selected_features = X_train.columns[(feature_sel_model.get_support())]
print(selected_features)

# Print some stats
print('Total Features: {}'.format((X_train.shape[1])))
print('Selected Features: {}'.format(len(selected_features)))
print('Features with coefficients shrank to zero: {}'.format(np.sum(feature_sel_model.estimator_.coef_ == 0)))

X_train = X_train[selected_features]
print(X_train.head())
