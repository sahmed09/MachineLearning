import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv('../Datasets/titanic_train.csv')
print(df.head())
print(df['Age'].isnull().sum())

sns.histplot(df['Age'].dropna(), kde=True)
plt.show()

sns.histplot(df['Age'].fillna(100), kde=True)
plt.show()

# Gaussian (Normal) Distributed
print('\nIf The Data Is Normally Distributed')

figure = df.Age.hist(bins=50)
figure.set_title('Age')
figure.set_xlabel('Age')
figure.set_ylabel('No of passenger')
plt.show()

df.boxplot(column='Age')
plt.show()

print(df['Age'].describe())

# If The Data Is Normally Distributed We use this
# Assuming Age follows A Gaussian Distribution, we will calculate the boundaries which differentiates the outliers
upper_boundary = df['Age'].mean() + 3 * df['Age'].std()
lower_boundary = df['Age'].mean() - 3 * df['Age'].std()
print('Mean (Age):', df['Age'].mean())
print('Lower Boundary (Age):', lower_boundary)
print('Upper Boundary (Age):', upper_boundary)

# If Features Are Skewed We Use the below Technique
print('\nIf Features Are Skewed')

figure = df.Fare.hist(bins=50)
figure.set_title('Fare')
figure.set_xlabel('Fare')
figure.set_ylabel('No of passenger')
plt.show()

df.boxplot(column='Fare')
plt.show()

print(df['Fare'].describe())

# Lets compute the Interquartile range to calculate the boundaries
# In case of skewed data, it is not sure that the IQR range technique will work. We have to verify it
IQR_fare = df.Fare.quantile(0.75) - df.Fare.quantile(0.25)
print('IQR of Fare:', IQR_fare)

lower_bridge_fare = df['Fare'].quantile(0.25) - (IQR_fare * 1.5)
upper_bridge_fare = df['Fare'].quantile(0.75) + (IQR_fare * 1.5)
print('Lower Bridge_Fare:', lower_bridge_fare)
print('Upper Bridge_Fare:', upper_bridge_fare)

# Extreme outliers (When data is skewed, we calculate the extreme outliers)
lower_bridge_fare_extreme = df['Fare'].quantile(0.25) - (IQR_fare * 3)
upper_bridge_fare_extreme = df['Fare'].quantile(0.75) + (IQR_fare * 3)
print('Lower Bridge_Fare (Extreme outliers):', lower_bridge_fare_extreme)
print('Upper Bridge_Fare (Extreme outliers):', upper_bridge_fare_extreme)

data = df.copy()

data.loc[data['Age'] >= 73, 'Age'] = 73  # Replacing where the data['Age'] >= 73 with 73
figure = data.Age.hist(bins=50)
figure.set_title('Age')
figure.set_xlabel('Age')
figure.set_ylabel('No of passenger')
plt.show()

data.loc[data['Fare'] >= 100, 'Fare'] = 100  # Replacing where the data['Fare'] >= 100 with 100
figure = data.Fare.hist(bins=50)
figure.set_title('Fare')
figure.set_xlabel('Fare')
figure.set_ylabel('No of passenger')
plt.show()

print('\nMachine Learning Models')
X_train, X_test, y_train, y_test = train_test_split(data[['Age', 'Fare']].fillna(0), data['Survived'], test_size=0.3)

# Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)

print('Accuracy_score (LogisticRegression): {}'.format(accuracy_score(y_test, y_pred)))
print('roc_auc_score (LogisticRegression): {}'.format(roc_auc_score(y_test, y_pred_prob[:, 1])))

# RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)

print('Accuracy_score (RandomForestClassifier): {}'.format(accuracy_score(y_test, y_pred)))
print('roc_auc_score (RandomForestClassifier): {}'.format(roc_auc_score(y_test, y_pred_prob[:, 1])))
