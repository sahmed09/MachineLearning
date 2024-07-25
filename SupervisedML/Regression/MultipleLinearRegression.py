import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

df_index = pd.read_csv('../../Datasets/economic_index.csv')
print(df_index.head())

# Drop unnecessary columns
df_index.drop(columns=['Unnamed: 0', 'year', 'month'], axis=1, inplace=True)  # inplace=True, Actually dropping the columns
print(df_index.head())

# Check null values
print(df_index.isnull().sum())

# Visualization
# sns.pairplot(df_index)
# plt.show()

# Check Correlation
print(df_index.corr())

# Visualize the data points
# plt.scatter(df_index['interest_rate'], df_index['unemployment_rate'], color='r')
# plt.xlabel('Interest Rate')
# plt.ylabel('Unemployment Rate')
# plt.show()

# Independent and Dependent Features
# X = df_index[['interest_rate', 'unemployment_rate']]  # Independent Features
X = df_index.iloc[:, :-1]  # Independent Features
y = df_index.iloc[:, -1]  # Dependent Features
print(X.head())
print(y.head())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Plot data and a linear regression model fit
# sns.regplot(data=df_index, x='interest_rate', y='index_price')
# sns.regplot(data=df_index, x='interest_rate', y='unemployment_rate')
# sns.regplot(data=df_index, x='index_price', y='unemployment_rate')
# plt.show()

# Standardization
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
print(X_test)

# Apply Simple Linear Regression
l_regression = LinearRegression()
l_regression.fit(X_train, y_train)
print("Co-efficient or Slope:", l_regression.coef_)
print("Intercept:", l_regression.intercept_)

# Cross Validation
validation_score = cross_val_score(l_regression, X_train, y_train, scoring='neg_mean_squared_error', cv=3)
print('Validation Score:', validation_score)
print('Mean of Validation Score:', np.mean(validation_score))

# Prediction
y_prediction = l_regression.predict(X_test)
print('y-prediction:', y_prediction)

# Performance Metrics
mse = mean_squared_error(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)
rmse = np.sqrt(mse)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

print("No of Independent Features:", X_test.shape[1])
score = r2_score(y_test, y_prediction)
adjusted_r_squared = 1 - ((1 - score) * (len(y_test) - 1)) / (len(y_test) - X_test.shape[1] - 1)
print("R squared:", score)
print("Adjusted R squared:", adjusted_r_squared)

# Prediction for new Data
new_prediction = l_regression.predict(scalar.transform([[3.5, 4.5]]))
print('New Prediction:', new_prediction)

# Assumptions
# plt.scatter(y_test, y_prediction)
# plt.show()

residuals = y_test - y_prediction
print(residuals)

# Plot Residuals
sns.displot(residuals, kind='kde')
plt.show()

# Scatter plot with respect to predictions and residuals
# plt.scatter(y_prediction, residuals)
# plt.show()  # Data is uniformly distributed which is good, if it follow any pattern -> something is wrong

# Ordinary Least Square (OLS) Linear Regression
model = sm.OLS(y_train, X_train).fit()
prediction = model.predict(X_test)
print(prediction)
print(model.summary())
