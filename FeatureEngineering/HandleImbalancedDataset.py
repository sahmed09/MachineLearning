import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from imblearn.under_sampling import NearMiss  # UnderSampling
from imblearn.combine import SMOTETomek  # OverSampling
from imblearn.over_sampling import RandomOverSampler  # OverSampling
from collections import Counter
# pip install imbalanced-learn

rcParams['figure.figsize'] = 12, 6
RANDOM_SEED = 42
LABELS = ['Normal', 'Fraud']

print('Handle Imbalanced Dataset - UnderSampling')
# UnderSampling: Not wise to use - only use in case of small dataset

data = pd.read_csv('../Datasets/creditcard_fraud.csv')  # Class 0 = Normal Transaction, Class 1 = Fraud Transaction
print(data.shape)
print(data.head())
print(data.info())
print(data['Class'].value_counts())

# Create independent and Dependent Features
X = data.drop(labels=['Class'], axis=1)
y = data['Class']
print(X.shape, y.shape)

state = np.random.RandomState(42)
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
print(X_outliers.shape)

# Exploratory Data Analysis
print('Check Null Values:', data.isnull().values.any())

count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Get the Fraud and the normal dataset
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]
print('Fraud:', fraud.shape, 'Normal:', normal.shape)

# Implementing UnderSampling for Handling Imbalanced Dataset
nm = NearMiss()
X_res, y_res = nm.fit_resample(X, y)
print(X_res.shape, y_res.shape)

print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))

print('\nHandle Imbalanced Dataset - OverSampling')

data = pd.read_csv('../Datasets/creditcard_fraud.csv')  # Class 0 = Normal Transaction, Class 1 = Fraud Transaction
print(data.shape)
print(data.head())
print(data.info())

# Create independent and Dependent Features
X = data.drop(labels=['Class'], axis=1)
y = data['Class']
print(X.shape, y.shape)

state = np.random.RandomState(42)
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
print(X_outliers.shape)

# Exploratory Data Analysis
print('Check Null Values:', data.isnull().values.any())

count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Get the Fraud and the normal dataset
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]
print('Fraud:', fraud.shape, 'Normal:', normal.shape)

# Implementing OverSampling for Handling Imbalanced Dataset
# SMOTETomek is a hybrid method which uses an under sampling method (Tomek) in with an over sampling method (SMOTE).
smk = SMOTETomek(random_state=42, sampling_strategy=0.5)
X_res, y_res = smk.fit_resample(X, y)
print(X_res.shape, y_res.shape)

print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape (SMOTETomek: ratio=0.5) {}'.format(Counter(y_res)))

os_us = SMOTETomek(random_state=42, sampling_strategy=1)
X_train_res1, y_train_res1 = os_us.fit_resample(X, y)
print(X_train_res1.shape, y_train_res1.shape)

print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape (SMOTETomek: ratio=1) {}'.format(Counter(y_train_res1)))

# RandomOverSampler to handle imbalanced data
os = RandomOverSampler(random_state=42, sampling_strategy=0.5)
X_train_res, y_train_res = os.fit_resample(X, y)
print(X_train_res.shape, y_train_res.shape)

print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape (RandomOverSampler) {}'.format(Counter(y_train_res)))
