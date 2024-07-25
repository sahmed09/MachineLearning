import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Lasso  # for feature selection
from sklearn.feature_selection import SelectFromModel  # for feature selection

df = pd.read_csv('../Datasets/mobile_dataset.csv')
print(df.head())
print(df.shape)

# Univariate Selection
X = df.iloc[:, :-1]
y = df['price_range']
print(X.head())
print(y.head())

"""Option 1: Apply SelectKBest Algorithm"""
print('\nOption 1: Apply SelectKBest Algorithm')
ordered_rank_features = SelectKBest(score_func=chi2, k=20)
ordered_feature = ordered_rank_features.fit(X, y)

df_scores = pd.DataFrame(ordered_feature.scores_, columns=['Score'])
df_columns = pd.DataFrame(X.columns)

features_rank = pd.concat([df_columns, df_scores], axis=1)
features_rank.columns = ['Features', 'Score']
print(features_rank)

print(features_rank.nlargest(10, 'Score'))

"""Option 2: Feature Importance"""
print('\nOption 2: Feature Importance')
# This technique gives a score for each feature of data,the higher the score the more relevant it is
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)

ranked_features = pd.Series(model.feature_importances_, index=X.columns)
ranked_features.nlargest(10).plot(kind='barh')
plt.show()

"""Option 3: Correlation"""
print('\nOption 3: Correlation')
print(df.corr())

# Plotting multicollinearity
corr = df.iloc[:, :-1].corr()
top_features = corr.index
plt.figure(figsize=(20, 20))
sns.heatmap(df[top_features].corr(), annot=True)
plt.show()


# Check for multicollinearity
# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                col_name = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(col_name)
    return col_corr


threshold = 0.8
corr_features = correlation(df.iloc[:, :-1], threshold)
print(corr_features)

"""Option 4: Information Gain"""
print('\nOption 4: Information Gain')
mutual_info = mutual_info_classif(X, y)
print(mutual_info)

mutual_data = pd.Series(mutual_info, index=X.columns)
print(mutual_data.sort_values(ascending=False))

"""Option 5: Using Lasso Regression and SelectFromModel"""
print('\nOption 5: Using Lasso Regression and SelectFromModel')
dataset = pd.read_csv('../Datasets/House_Price_X_train.csv')
X_train = dataset.drop(['Id', 'SalePrice'], axis=1)
y_train = dataset[['SalePrice']]

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(X_train, y_train)
print(feature_sel_model.get_support())

selected_features = X_train.columns[(feature_sel_model.get_support())]
print(selected_features)

print('Total Features: {}'.format((X_train.shape[1])))
print('Selected Features: {}'.format(len(selected_features)))
print('Features with coefficients shrank to zero: {}'.format(np.sum(feature_sel_model.estimator_.coef_ == 0)))

X_train = X_train[selected_features]
print(X_train.head())
