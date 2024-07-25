import pandas as pd
import statsmodels.api as sm

print('This example will show no multicollinearity among independent variables')

df_adv = pd.read_csv('../../Datasets/Advertising.csv', index_col=0)
print(df_adv.head())

X = df_adv[['TV', 'radio', 'newspaper']]  # Independent Feature
y = df_adv['sales']  # Dependent Feature

X = sm.add_constant(X)
print(X)

# fit a OLS model with intercept on TV and Radio
model = sm.OLS(y, X).fit()
print(model.summary())
# In case of no multicollinearity, std_err will be less, also the correlation is low

print(X.iloc[:, 1:].corr())
print()

print('This example will show multicollinearity among independent variables')

df_salary = pd.read_csv('../../Datasets/Salary_Data.csv')
print(df_salary.head())

X = df_salary[['YearsExperience', 'Age']]  # Independent Feature
y = df_salary['Salary']  # Dependent Feature

# fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
print(X.head())

model = sm.OLS(y, X).fit()
print(model.summary())
# In case of multicollinearity, std_err will be very high, also the correlation is high (98.75%)

print(X.iloc[:, 1:].corr())
# To solve multicollinearity issue, drop the 'Age' Feature as it has higher P value than 'YearsExperience'
