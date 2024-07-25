import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm  # For using OLS (Ordinary Least Square) Linear Regression

df = pd.read_csv('../../Datasets/height-weight.csv')
print(df.head())

# Scatter Plot
# plt.scatter(df['Weight'], df['Height'])
# plt.xlabel('Weight')
# plt.ylabel('Height')
# plt.show()

# Correlation
print(df.corr())

# sns.pairplot(df)
# plt.show()

# Independent and Dependent Feature
X = df[['Weight']]  # Independent Features should be in dataframe or 2 Dimensional Array
y = df['Height']  # Dependent feature can be in series form or 1-D array
# print(X)
# print(type(X))
print(X.shape)
# print(np.array(X))
# print(np.array(X).shape)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)

# Standardization  # Take all independent features and Apply z_score and convert all values using mean=0,
# standard_deviation=1. z_score = (xi-mean)/standard_deviation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # In training, we use fit_transform
# print(X_train)
X_test = scaler.transform(X_test)
# In testing, we use only transform. means we use the mean and standard deviation of training data in test data
# print(X_test)

# Apply Simple Linear Regression
l_regression = LinearRegression()
l_regression.fit(X_train, y_train)
print("Co-efficient or Slope:", l_regression.coef_)
print("Intercept:", l_regression.intercept_)
print('Variance score: {}'.format(l_regression.score(X_test, y_test)))

# Plot Best fit line with respect to training data
# plt.scatter(X_train, y_train)
# plt.plot(X_train, l_regression.predict(X_train))
# plt.show()

# Prediction for test data
"""1. Predicted height output = intercept + coef_(Weights)
2. y_pred_test = 156.470 + 17.29(X_test)"""
y_prediction = l_regression.predict(X_test)

plt.scatter(X_train, y_train, color="green")   
plt.plot(X_train, l_regression.predict(X_train), color="red")    
plt.title('Height vs Weight (Training Dataset)')
plt.xlabel("Height")  
plt.ylabel("Weight")  
plt.show()   

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
new_prediction = l_regression.predict([[72]])  # It will give non standardized output. Need to use Standardization
print("Without Scaling:", new_prediction)

new_prediction = l_regression.predict(scaler.transform([[72]]))
print("After Scaling:", new_prediction)

# plot for residual error
# setting plot style
plt.style.use('fivethirtyeight')
# plotting residual errors in training data
plt.scatter(l_regression.predict(X_train),
            l_regression.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

# plotting residual errors in test data
plt.scatter(l_regression.predict(X_test),
            l_regression.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

# plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper left')
plt.title("Residual errors")
plt.show()

# Ordinary Least Square (OLS) Linear Regression
model = sm.OLS(y_train, X_train).fit()
prediction = model.predict(X_test)
print(prediction)
print(model.summary())
