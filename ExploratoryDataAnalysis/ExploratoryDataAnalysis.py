import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

train = pd.read_csv('Datasets/titanic_train.csv')
print(train.head())
print(train.columns)  # SibSp -> Count of siblings and spouse, Parch -> Count to parent and children

"""Exploratory Data Analysis"""

"""Missing Data"""
print(train.isnull())
# use seaborn to create a simple heatmap to see where we are missing data
# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# plt.show()

# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable
# replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much
# of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another
# feature like "Cabin Known: 1 or 0"

# sns.set_style('whitegrid')
# sns.countplot(x='Survived', data=train, hue='Survived')
# plt.show()

# sns.set_style('whitegrid')
# sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r')
# plt.show()

# sns.set_style('whitegrid')
# sns.countplot(x='Survived', data=train, hue='Pclass', palette='rainbow')
# plt.show()

# See distribution of Age
# sns.histplot(train['Age'].dropna(), kde=False, color='darkred', bins=40)
# plt.show()

# plt.hist(train['Age'], color='darkred', bins=30, alpha=0.3)
# plt.show()

# sns.countplot(x='SibSp', data=train, hue='SibSp')
# plt.show()

# plt.hist(train['Fare'], color='green', bins=40)
# plt.show()

"""Data Cleaning"""
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by
# filling in the mean age of all the passengers (imputation). However we can be smarter about this and check the
# average age by passenger class.
# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass', y='Age', data=train, palette='winter', hue='Pclass')
# plt.show()

# We can see the wealthier passengers in the higher classes tend to be older, which makes sense.
# We'll use these average age values to impute based on Pclass for Age.


def impute_age(cols):
    age = cols[0]
    p_class = cols[1]

    if pd.isnull(age):
        if p_class == 1:
            return 37
        elif p_class == 2:
            return 29
        else:
            return 24
    else:
        return age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# plt.show()

# Now drop the Cabin column and the row in Embarked that is NaN.
train.drop('Cabin', axis=1, inplace=True)
print(train.head())
train.dropna(inplace=True)
print(train)

# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# plt.show()

"""Converting Categorical Features
We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm 
won't be able to directly take in those features as inputs."""
print(train.info())
print(pd.get_dummies(train['Embarked'], drop_first=True).head())

Sex = pd.get_dummies(train['Sex'], drop_first=True)
Embarked = pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
print(train.head())

train = pd.concat([train, Sex, Embarked], axis=1)
print(train.head())

"""Building a Logistic Regression model"""
print(train.drop('Survived', axis=1).head())
print(train['Survived'].head())

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1), train['Survived'],
                                                    test_size=0.30, random_state=101)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

predictions = log_model.predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions)
print(conf_matrix)

accuracy = accuracy_score(y_test, predictions)
print(accuracy)

print(predictions)

report = classification_report(y_test, predictions)
print(report)
