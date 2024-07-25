import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat
import pylab
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression

titanic = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Pclass', 'Age', 'Fare', 'Survived'])
print(titanic.head())
print(titanic.isnull().sum())

titanic['Age'].fillna(titanic.Age.median(), inplace=True)
print(titanic.isnull().sum())

X = titanic.iloc[:, 1:]
y = titanic.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classification = LogisticRegression()
classification.fit(X_train_scaled, y_train)
y_pred = classification.predict(X_test)

min_max = MinMaxScaler()
df_minmax = pd.DataFrame(min_max.fit_transform(X_train))
print(df_minmax.head())

scaler = RobustScaler()
df_robust_scaler = pd.DataFrame(scaler.fit_transform(X_train))
print(df_robust_scaler.head())
X_test_robust_scaler = scaler.transform(X_test)


# Q-Q Plot (to check whether feature is guassian or normal distributed)
def plot_data(df, feature):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    titanic[feature].hist()
    plt.subplot(1, 2, 2)
    stat.probplot(titanic[feature], dist='norm', plot=pylab)
    plt.show()


plot_data(titanic, 'Age')
