import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""What are the different types of Missing Data?
1. Missing Completely at Random (MCAR)"""

df = pd.read_csv('../Datasets/titanic_train.csv')
print(df.head())
print(df.isnull().sum())

print('Missing Completely at Random (MCAR):\n', df[df['Embarked'].isnull()])

"""
2. Missing Data Not At Random (MNAR): 
Systematic missing Values. There is absolutely some relationship between the data missing and any other values, 
observed or missing, within the dataset."""
print('\nMissing Data Not At Random (MNAR)')

print(df.columns)

df['cabin_null'] = np.where(df['Cabin'].isnull(), 1, 0)
print('Mean of Cabin Null:', df['cabin_null'].mean())  # find the percentage of null values

print(df.groupby(['Survived'])['cabin_null'].mean())

"""All the techniques of handling missing values"""

"""1. Mean/Median/Mode Imputation
When should we apply? Mean/median imputation has the assumption that the data are missing completely at random (MCAR). 
We solve this by replacing the NAN with the most frequent occurrence of the variables"""
print('\nMean/Median/Mode Imputation')

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Age', 'Fare', 'Survived'])
print(df.head())

# percentage of missing values
print(df.isnull().mean())


def impute_nan(df, variable, median):
    df[variable + '_median'] = df[variable].fillna(median)


median = df.Age.median()
print('Median of Age:', median)

impute_nan(df, 'Age', median)
print(df.head())

print('Standard Deviation of Age:', df['Age'].std())
print('Standard Deviation of Age_median:', df['Age_median'].std())

fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.show()

"""2. Random Sample Imputation
Aim: Random sample imputation consists of taking random observation from the dataset and we use this observation 
to replace the nan values.
When should it be used? It assumes that the data are missing completely at random (MCAR)"""
print('\nRandom Sample Imputation')

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Age', 'Fare', 'Survived'])
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.isnull().mean())
print('Total Null values in Age:', df['Age'].isnull().sum())

print(df['Age'][689])

# Dropping the nan values and picking up a random sample from the remaining values to fill up the missing values
print(df['Age'].dropna().sample(df['Age'].isnull().sum(), random_state=0))  # replace null values with a specific value
print(df[df['Age'].isnull()].index)


def impute_nan(df, variable, median):
    df[variable + '_median'] = df[variable].fillna(median)
    df[variable + '_random'] = df[variable]  # Copying the variable values

    # It will have the random sample to fill the nan values
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)

    # pandas need to have same index in order to merge the dataset
    random_sample.index = df[df[variable].isnull()].index
    # where ever the value is null in the variable+'_random' feature, replace it with the 'random_sample'
    df.loc[df[variable].isnull(), variable + '_random'] = random_sample


median = df.Age.median()
print('Median of Age:', median)

impute_nan(df, 'Age', median)
print(df.head(20))

fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind='kde', ax=ax, color='red')
df.Age_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.show()

"""3. Capturing NAN values with a new feature
It works well if the data are not missing completely at random"""
print('\nCapturing NAN values with a new feature')

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Age', 'Fare', 'Survived'])
print(df.head())

df['Age_NAN'] = np.where(df['Age'].isnull(), 1, 0)
print(df.head(20))

print('Median of Age:', df.Age.median())

df['Age'].fillna(df.Age.median(), inplace=True)
print(df.head(20))

"""4. End of Distribution Imputation
In this distribution, we will try to take the end of the distribution and will try to replace it."""
print('\nEnd of Distribution Imputation')

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Age', 'Fare', 'Survived'])
print(df.head())

df.Age.hist(bins=50)
plt.show()

# Take data after the 3rd standard deviation from the mean
extreme = df.Age.mean() + 3 * df.Age.std()
print(extreme)

sns.boxplot(data=df, x='Age')
plt.show()


def impute_nan(df, variable, median, extreme):
    df[variable + '_end_distribution'] = df[variable].fillna(extreme)
    df[variable].fillna(median, inplace=True)


median = df.Age.median()
print('Median of Age:', median)

impute_nan(df, 'Age', median, extreme)

print(df.head(20))

df.Age.hist(bins=50)
plt.show()

df['Age_end_distribution'].hist(bins=50)
plt.show()

sns.boxplot(data=df, x='Age_end_distribution')
plt.show()

"""5. Arbitrary Value Imputation -> Not important
This technique was derived from kaggle competition. It consists of replacing NAN by an arbitrary value."""
print('\nArbitrary Value Imputation')

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Age', 'Fare', 'Survived'])
print(df.head(10))


def impute_nan(df, variable):
    df[variable + '_zero'] = df[variable].fillna(0)
    df[variable + '_hundred'] = df[variable].fillna(100)


impute_nan(df, 'Age')
print(df.head(20))

df['Age'].hist(bins=50)
plt.show()

"""How To Handle Categorical Missing Values
6. Frequent Categories Imputation"""
print('\nFrequent Categories Imputation')

df = pd.read_csv('../Datasets/house-price-train.csv', usecols=['BsmtQual', 'FireplaceQu', 'GarageType', 'SalePrice'])
print(df.head())
print(df.columns)
print(df.shape)
print(df.isnull().sum())
print(df.isnull().mean().sort_values(ascending=True))

# Compute the frequency with every feature
print(df.groupby(['BsmtQual'])['BsmtQual'].count())

df['BsmtQual'].value_counts().plot.bar()
plt.show()

df.groupby(['BsmtQual'])['BsmtQual'].count().sort_values(ascending=False).plot.bar()
plt.show()

df['GarageType'].value_counts().plot.bar()
plt.show()

df['FireplaceQu'].value_counts().plot.bar()
plt.show()

print('Categories (GarageType)', df['GarageType'].value_counts().index)  # Category names
print(df['GarageType'].value_counts())
print('Most Frequent Category (GarageType) :', df['GarageType'].value_counts().index[0])
print(df['GarageType'].mode())  # Most Frequent Category
print('Most Frequent Category (GarageType) :', df['GarageType'].mode()[0])


def impute_nan(df, variable):
    most_frequent_category = df[variable].mode()[0]
    df[variable].fillna(most_frequent_category, inplace=True)


for feature in ['BsmtQual', 'FireplaceQu', 'GarageType']:
    impute_nan(df, feature)

print(df.head())
print(df.isnull().mean())

"""Adding a variable to capture NAN"""
print('\nAdding a variable to capture NAN')

df = pd.read_csv('../Datasets/house-price-train.csv', usecols=['BsmtQual', 'FireplaceQu', 'GarageType', 'SalePrice'])
print(df.head())

df['BsmtQual_Var'] = np.where(df['BsmtQual'].isnull(), 1, 0)
print(df.head())

print('Most Frequent Category (BsmtQual) :', df['BsmtQual'].mode()[0])

frequent = df['BsmtQual'].mode()[0]
df['BsmtQual'].fillna(frequent, inplace=True)
print(df.head())

df['FireplaceQu_Var'] = np.where(df['FireplaceQu'].isnull(), 1, 0)
frequent = df['FireplaceQu'].mode()[0]
print('Most Frequent Category (FireplaceQu) :', frequent)
df['FireplaceQu'].fillna(frequent, inplace=True)
print(df.head())

"""If you have more frequent categories, we just replace NAN with a new category
Instead of taking the most frequent category, we just say that NAN is a another category itself 
(take it as missing value)"""
print('\nReplace NAN with a new category')

df = pd.read_csv('../Datasets/house-price-train.csv', usecols=['BsmtQual', 'FireplaceQu', 'GarageType', 'SalePrice'])
print(df.head())


def impute_nan(df, variable):
    df[variable + 'newvar'] = np.where(df[variable].isnull(), 'Missing', df[variable])


for feature in ['BsmtQual', 'FireplaceQu', 'GarageType']:
    impute_nan(df, feature)

print(df.head())

df = df.drop(['BsmtQual', 'FireplaceQu', 'GarageType'], axis=1)
print(df.head())

# def impute_nan(df, variable):
#     """Imputes missing values in the 'Age' column using median and a fallback strategy."""
#
#     # Check for missing values
#     if df[variable].isnull().any():
#         # Calculate median for imputation
#         median = df[variable].median()
#
#         # Handle potential zero standard deviation (fallback)
#         try:
#             # Attempt imputation using mean and standard deviation
#             extreme = df[variable].mean() + 3 * df[variable].std()
#             df[variable + '_end_distribution'] = df[variable].fillna(extreme)
#         except ZeroDivisionError:
#             # Fallback strategy if standard deviation is zero
#             print("Standard deviation is zero. Using median for imputation on '{}'.".format(variable))
#             df[variable + '_end_distribution'] = df[variable].fillna(median)
#
#         # Impute remaining missing values with median
#         df[variable].fillna(median, inplace=True)
#     else:
#         print("No missing values found in '{}'.".format(variable))
