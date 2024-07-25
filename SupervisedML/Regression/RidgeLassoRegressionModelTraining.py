import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('Datasets/Algerian_forest_fires_dataset_CLEAN.csv')
print(df.head())
print(df.columns)

# drop month,day and year
df.drop(['day', 'month', 'year'], axis=1, inplace=True)
print(df.head())
print(df['Classes'].value_counts())

# Encoding
df['Classes'] = np.where(df['Classes'].str.contains('not fire'), 0, 1)
print(df.tail())
print(df['Classes'].value_counts())

# Independent And dependent features
X = df.drop('FWI', axis=1)
y = df['FWI']
print(X.head())
print(y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape, X_test.shape)

# Feature Selection based on correlaltion
print(X_train.corr())

# Plotting multicollinearity
plt.figure(figsize=(12, 10))
corr = X_train.corr()
sns.heatmap(corr, annot=True)
plt.show()


# Check for multicollinearity
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# threshold--Domain expertise
corr_features = correlation(X_train, 0.85)
print(corr_features)

# drop features when correlation is more than 0.85
X_train.drop(corr_features, axis=1, inplace=True)
X_test.drop(corr_features, axis=1, inplace=True)
print(X_train.shape, X_test.shape)

"""Feature Scaling Or Standardization"""
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)
print(X_train_scaled)

# Box Plots To understand Effect Of Standard Scaler
plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=X_train)
plt.title('X_train Before Scaling')
plt.subplot(1, 2, 2)
sns.boxplot(data=X_train_scaled)
plt.title('X_train After Scaling')
plt.show()

"""Linear Regression Model"""
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)
y_prediction = linear_reg.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_prediction)
score = r2_score(y_test, y_prediction)
print('LR Mean Absolute Error:', mae)
print('LR R2 Score:', score)
# plt.scatter(y_test, y_prediction)
# plt.show()

"""Lasso Regression"""
# lasso = Lasso(alpha=0.1)
lasso = Lasso()
lasso.fit(X_train_scaled, y_train)
y_prediction = lasso.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_prediction)
score = r2_score(y_test, y_prediction)
print('Lasso Mean Absolute Error:', mae)
print('Lasso R2 Score:', score)

plt.scatter(y_test, y_prediction)
plt.show()

# Cross Validation Lasso
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train_scaled, y_train)
print(lasso_cv.alpha_)
print(lasso_cv.alphas_)
# print(lasso_cv.mse_path_)

y_pred = lasso_cv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print('LassoCV Mean Absolute Error:', mae)
print('LassoCV R2 Score:', score)
print(lasso_cv.score(X_train_scaled, y_train))
print(lasso_cv.score(X_test_scaled, y_test))

plt.scatter(y_test, y_pred)
plt.show()

"""Ridge Regression"""
# ridge = Ridge(alpha=0.1)
ridge = Ridge()
ridge.fit(X_train_scaled, y_train)
y_prediction = ridge.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_prediction)
score = r2_score(y_test, y_prediction)
print('Ridge Mean Absolute Error:', mae)
print('Ridge R2 Score:', score)

plt.scatter(y_test, y_prediction)
plt.show()

# Cross Validation Ridge
ridge_cv = RidgeCV(cv=5)
ridge_cv.fit(X_train_scaled, y_train)
y_pred = ridge_cv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print('RidgeCV Mean Absolute Error:', mae)
print('RidgeCV R2 Score:', score)
print(ridge_cv.get_params())
print(ridge_cv.gcv_mode)

plt.scatter(y_test, y_pred)
plt.show()

"""Elasticnet Regression"""
elastic = ElasticNet()
elastic.fit(X_train_scaled, y_train)
y_prediction = elastic.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_prediction)
score = r2_score(y_test, y_prediction)
print('Elasticnet Mean Absolute Error:', mae)
print('Elasticnet R2 Score:', score)

plt.scatter(y_test, y_prediction)
plt.show()

# Cross Validation Elasticnet
elastic_cv = ElasticNetCV(cv=5)
elastic_cv.fit(X_train_scaled, y_train)
y_pred = elastic_cv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print('ElasticnetCV Mean Absolute Error:', mae)
print('ElasticnetCV R2 Score:', score)
print(elastic_cv.alpha_)
print(elastic_cv.alphas_)

plt.scatter(y_test, y_pred)
plt.show()
