import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

"""Feature Engineering Steps ->
1. Missing Values
2. Temporal variables
3. Categorical variables: remove rare labels
4. Feature Scaling (Standardise the values of the variables to the same range)"""

# pd.set_option('display.max_columns', None)  # to visualize all the columns of the dataframe
# pd.set_option('display.max_rows', None)  # to visualize all the rows of the dataframe

dataset = pd.read_csv('../Datasets/house-price-train.csv')
print(dataset.head())
print(dataset.dtypes)

"""# Due to the chance of data leakage, we need to split the data first and then apply feature engineering.
# Apply the feature engineering to the train data, then apply the same feature engineering to the test data.
# In this case, the train and test datasets are separate, so we can skip this step.
X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['SalePrice'], test_size=0.1, random_state=0)
print(X_train.shape, X_test.shape)"""

# TODO: 1. Missing Values
# Capture all the nan values
# First lets handle Categorical features which are missing -> by default categorical feature has Object('O') datatype
features_nan = [feature for feature in dataset.columns if
                dataset[feature].isnull().sum() > 1 and dataset[feature].dtypes == 'O']
for feature in features_nan:
    print('{}: {}% missing values'.format(feature, np.round(dataset[feature].isnull().mean(), 2)))


# Replace missing values with a new label
def replace_categorical_features(dataset, features_nan):
    data = dataset.copy()
    data[features_nan] = data[features_nan].fillna('Missing')
    return data


dataset = replace_categorical_features(dataset, features_nan)
print(dataset[features_nan].isnull().sum())
print(dataset.head())

# Now check for numerical variables the contains missing values
numerical_with_nan = [feature for feature in dataset.columns if
                      dataset[feature].isnull().sum() > 1 and dataset[feature].dtypes != 'O']
print(numerical_with_nan)

for feature in numerical_with_nan:
    print('{}: {}% missing values'.format(feature, np.round(dataset[feature].isnull().mean(), 2)))

# Replacing the numerical Missing Values
for feature in numerical_with_nan:
    # We will replace by using median (we can use mode also) since there are outliers
    median_value = dataset[feature].median()

    # create a new feature to capture nan values
    dataset[feature + 'nan'] = np.where(dataset[feature].isnull(), 1, 0)
    dataset[feature].fillna(median_value, inplace=True)

print(dataset[numerical_with_nan].isnull().sum())
print(dataset.head(30))

# print(dataset['GarageYrBlt'].head())
# print(dataset['GarageYrBlt'].dtypes)

# TODO: 2. Temporal variables (Date Time Variables)
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:  # found while doing EDA
    dataset[feature] = dataset['YrSold'] - dataset[feature]

print(dataset.head())
print(dataset[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head())

# Numerical Variables
# Since the numerical variables are skewed (not in the form of gaussian distribution), we will perform
# log normal distribution.
# These features don't have 0 values inside it.
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

# print(dataset['LotFrontage'].head())

for feature in num_features:
    dataset[feature] = np.log(dataset[feature])
print(dataset.head())
# print(dataset['LotFrontage'].head())

# TODO: 3. Categorical variables: remove rare labels
# Handling Rare Categorical Feature
# We will remove categorical variables that are present less than 1% of the observations
categorical_features = [feature for feature in dataset.columns if dataset[feature].dtypes == 'O']
print(categorical_features)

for feature in categorical_features:
    temp = dataset.groupby(feature)['SalePrice'].count() / len(dataset)
    temp_df = temp[temp > 0.01].index
    dataset[feature] = np.where(dataset[feature].isin(temp_df), dataset[feature], 'Rare_var')
    # taking the dataset[feature] if it is in temp_df index, otherwise converting it into a new label 'Rare_var'

print(dataset.head(100))
print(dataset['MSZoning'][88], dataset['MSZoning'][93])

# TODO: Featuring engineering on categorical features (converting categorical features into the numerical values)
for feature in categorical_features:
    labels_ordered = dataset.groupby(feature)['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
    dataset[feature] = dataset[feature].map(labels_ordered)

print(dataset.head())

# TODO: 4. Feature Scaling (Standardise the values of the variables to the same range)
feature_scale = [feature for feature in dataset.columns if feature not in ['Id', 'SalePrice']]
print(feature_scale)

scaler = MinMaxScaler()
scaler.fit(dataset[feature_scale])

print(scaler.transform(dataset[feature_scale]))
print(dataset['SalePrice'].head())
print(dataset['MSSubClass'].head())

# transform the train and test set, and add on the Id and SalePrice variables
data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                  pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)], axis=1)
# No need to apply scaler in 'Id' and 'SalePrice', so dropping them first and concatenating again after applying scaler
# scaler.transform(dataset[feature_scale]) -> returns values in array form and it needs to be converted in DataFrame
print(data.head())

data.to_csv('../Datasets/House_Price_X_train.csv', index=False)

# TODO: """DO THE SAME FEATURE ENGINEERING STEPS FOR TEST DATA ALSO."""
test_dataset = pd.read_csv('../Datasets/house-price-test.csv')
