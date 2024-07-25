import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data"""

"""Life Cycle of Machine Learning Projects
1. Data Analysis
2. Feature Engineering
3. Feature Selection
4. Model Building
5. Model Deployment"""

"""Exploratory Data Analysis (EDA) Phase -> Main aim is to understand more about the data
In Data Analysis we will analyze to find out the below stuff
1. Missing Values
2. All the numerical values
3. Distribution of the numerical values
4. Categorical Values
5. Cardinality of categorical Variables
6. Outliers
7. Relationship between independent and dependent features."""

# Display all the columns of the dataframe
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)  # Display all the rows

dataset = pd.read_csv('../Datasets/house-price-train.csv')
print(dataset.shape)  # Shape of dataset with rows and columns
print(dataset.head())

# TODO: 1. Missing Values
# Check the percentage of the nan values present in each feature
# Step - 1: Make the list of features which has missing values
features_with_nan = [features for features in dataset.columns if dataset[features].isnull().sum() > 1]
# Step - 2: Print the feature name and the percentage of missing values
for feature in features_with_nan:
    print(feature, np.round(dataset[feature].isnull().mean(), 4), ' % missing values')

# We need to find the relationship between missing values and SalePrice as there are many missing values
for feature in features_with_nan:
    data = dataset.copy()

    # Create a variable that indicates 1 if the observation wa missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)

    # Calculate the mean SalePrice where the information is missing or present
    # data.groupby(feature)['SalePrice'].median().plot.bar()
    # plt.title(feature)
    # plt.show()
    # We need to replace the nan values with something meaningful which will be done in the Feature Engineering Section.

print('Id of Houses {}'.format(len(dataset.Id)))

# TODO: 2. Numerical Variables
# List of numerical variables
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))

# Visualize the numerical variables
print(dataset[numerical_features].head())

# Temporal Variables (e.g. Datetime Variable) - 4 year variables
# List of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
print(year_feature)

# Explore the content of the year variables
for feature in year_feature:
    print(feature, dataset[feature].unique())

# Analyze the temporal Datetime variables
# Check whether there is a relationship between year the house is sold and SalePrice
# dataset.groupby('YrSold')['SalePrice'].median().plot()
# plt.xlabel('Year Sold')
# plt.ylabel('Median House Price')
# plt.title('House Price vs YearSold')
# plt.show()

# Compare the difference between All years feature and SalePrice
for feature in year_feature:
    if feature != 'YrSold':
        data = dataset.copy()
        # Capture the difference between year variable and year the house was sold for
        data[feature] = data['YrSold'] - data[feature]

        # plt.scatter(data[feature], data['SalePrice'])
        # plt.xlabel(feature)
        # plt.ylabel('SalePrice')
        # plt.show()

# TODO: 3. Distribution of the numerical values
# Numerical variables are usually of 2 types. 1. Continuous Variable, and 2. Discrete Variable
discrete_feature = [feature for feature in numerical_features if
                    len(dataset[feature].unique()) < 25 and feature not in year_feature + ['Id']]
print('Discrete Variables Count {}'.format(len(discrete_feature)))
print(discrete_feature)
print(dataset[discrete_feature].head())

# Find the relationship between them and SalePrice
for feature in discrete_feature:
    data = dataset.copy()
    # data.groupby(feature)['SalePrice'].median().plot().bar()
    grouped_data = data.groupby(feature)['SalePrice'].median()
    grouped_data.plot(kind='bar', x=grouped_data.index)  # Pass index as x-axis values
    # plt.xlabel(feature)
    # plt.ylabel('SalePrice')
    # plt.title(feature)
    # plt.show()

# Continuous Variables
continuous_feature = [feature for feature in numerical_features if
                      feature not in discrete_feature + year_feature + ['Id']]
print('Continuous feature Count {}'.format(len(continuous_feature)))
print(continuous_feature)

# Analyze the continuous values by creating histograms to understand the data
for feature in continuous_feature:
    data = dataset.copy()
    # data[feature].hist(bins=25)
    # plt.xlabel(feature)
    # plt.ylabel('Count')
    # plt.title(feature)
    # plt.show()

# Using Logarithmic Transformation
print([feature for feature in continuous_feature if 0 not in dataset[feature].unique()])
for feature in continuous_feature:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()

# TODO: 6. Outliers
for feature in continuous_feature:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        # data.boxplot(column=feature)
        # plt.ylabel(feature)
        # plt.title(feature)
        # plt.show()

# TODO: 4. Categorical Variables
categorical_features = [feature for feature in dataset.columns if dataset[feature].dtypes == 'O']
print(categorical_features)
print(dataset[categorical_features].head())

# TODO: 5. Cardinality of categorical Variables
# How many unique categories you have in your categorical features.
for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature, len(dataset[feature].unique())))

# TODO: 7. Relationship between categorical variables (independent) and dependent features
for feature in categorical_features:
    data = dataset.copy()
    grouped_data = data.groupby(feature)['SalePrice'].median()
    grouped_data.plot(kind='bar', x=grouped_data.index)  # Pass index as x-axis values
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
