import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat
import pylab
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

"""Standardization:
Standardization comes into picture when features of input data set have large differences between their ranges, 
or simply when they are measured in different measurement units (e.g., Pounds, Meters, Miles … etc).
We try to bring all the variables or features to a similar scale. Standardization means centering the variable at zero.
Data with respect to standard normal distribution -> use StandardScaler
z = (x - x_mean) / std"""
print('Standardization')

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Survived', 'Pclass', 'Age', 'Fare'])
print(df.head())
print(df.isnull().sum())

df['Age'].fillna(df.Age.median(), inplace=True)
print(df.isnull().sum())

# Independent and dependent features
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standardization: We use the StandardScaler from sklearn library
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=['Pclass', 'Age', 'Fare'])
print(X_train_scaled.head())
print(X_train_scaled.columns)

X_test_scaled = scaler.transform(X_test)
# print(X_test_scaled)

# Model Building
# fit() for training and predict for test
classification = LogisticRegression()
classification.fit(X_train_scaled, y_train)
y_pred = classification.predict(X_test_scaled)
print(y_pred)

plt.hist(X_train_scaled['Pclass'], bins=20)
# plt.hist(df_scaled[:, 1], bins=20)
plt.show()

plt.hist(X_train_scaled['Age'], bins=20)
# plt.hist(df_scaled[:, 2], bins=20)
plt.show()

plt.hist(X_train_scaled['Fare'], bins=20)
# plt.hist(df_scaled[:, 3], bins=20)
plt.show()

plt.hist(df['Fare'], bins=20)
plt.show()

"""Min Max Scaling (### CNN) --- Deep Learning Techniques
If data not maintains standard normal distribution -> use MinMaxScaler to normalize data
Min Max Scaling scales the values between 0 to 1. X_scaled = (X - X.min) / (X.max - X.min)"""
print('\nMin Max Scaling')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

min_max_scaler = MinMaxScaler()
df_min_max = min_max_scaler.fit_transform(X_train)
df_min_max = pd.DataFrame(df_min_max, columns=['Pclass', 'Age', 'Fare'])
print(df_min_max.head())

sns.pairplot(df_min_max)
plt.show()

plt.hist(df_min_max['Pclass'], bins=20)
plt.show()

plt.hist(df_min_max['Age'], bins=20)
plt.show()

plt.hist(df_min_max['Fare'], bins=20)
plt.show()

"""Robust Scaler
It is used to scale the feature to median and quantiles.
Scaling using median and quantiles consists of subtracting the median to all the observations, and then dividing by
the interquantile difference. The interquantile difference is the difference between the 75th and 25th quantile:
IQR = 75th quantile - 25th quantile
X_scaled = (X - X.median) / IQR"""
print('\nRobust Scaler')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

robust_scaler = RobustScaler()
df_robust_scaler = robust_scaler.fit_transform(X_train)
df_robust_scaler = pd.DataFrame(df_robust_scaler, columns=['Pclass', 'Age', 'Fare'])
print(df_robust_scaler.head())

X_test_robust_scaled = robust_scaler.transform(X_test)

sns.pairplot(df_robust_scaler)
plt.show()

plt.hist(df_robust_scaler['Age'], bins=20)
plt.show()

plt.hist(df_robust_scaler['Fare'], bins=20)
plt.show()

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Survived', 'Age', 'Fare'])
print(df.head())

# fillnan
df['Age'] = df['Age'].fillna(df['Age'].median())
print(df.isnull().sum())

sns.pairplot(df)
plt.show()


# If you want to check whether feature is gaussian or normal distributed
# Q-Q plot
def plot_data(df, feature):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df[feature].hist()
    plt.subplot(1, 2, 2)
    stat.probplot(df[feature], dist='norm', plot=pylab)
    plt.title('Probability Plot ' + '({})'.format(feature))
    plt.show()


plot_data(df, 'Age')

"""Logarithmic Transformation (not suitable for this dataset)"""
print('\nLogarithmic Transformation')

df['Age_log'] = np.log(df['Age'])
plot_data(df, 'Age_log')

"""Reciprocal Transformation"""
print('\nReciprocal Transformation')

df['Age_reciprocal'] = 1 / df.Age
plot_data(df, 'Age_reciprocal')

"""Square Root Transformation"""
print('\nSquare Root Transformation')

df['Age_square_root'] = df.Age ** (1 / 2)
plot_data(df, 'Age_square_root')

"""Exponential Transformation (Performs Better)"""
print('\nExponential Transformation')

df['Age_exponential'] = df.Age ** (1 / 1.2)
plot_data(df, 'Age_exponential')

"""Box Cox Transformation (Performs Better)"""
print('\nBox Cox Transformation')

df['Age_Boxcox'], parameters = stat.boxcox(df['Age'])
print(parameters)
plot_data(df, 'Age_Boxcox')

plot_data(df, 'Fare')

# Fare
df['Fare_log'] = np.log1p(df['Fare'])
plot_data(df, 'Fare_log')

df['Fare_Boxcox'], parameters = stat.boxcox(df['Fare'] + 1)
plot_data(df, 'Fare_Boxcox')
