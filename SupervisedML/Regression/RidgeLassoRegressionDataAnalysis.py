import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Datasets/Algerian_forest_fires_dataset_UPDATE.csv', header=1)
# print(dataset.head())
# print(dataset.info())

"""Basic Data Analysis"""
# Data Cleaning Process
# Missing Values
print(dataset[dataset.isnull().any(axis=1)])

"""The dataset is converted into two sets based on Region from 122th index, we can make a new column based on the Region
1 : "Bejaia Region Dataset"
2 : "Sidi-Bel Abbes Region Dataset"
Add new column with region"""
dataset.loc[:122, 'Region'] = 0
dataset.loc[122:, 'Region'] = 1
# print(dataset)
# print(dataset.info())
dataset[['Region']] = dataset[['Region']].astype(int)
# print(dataset.info())

# Removing the null values
# print(dataset.isnull().sum())
dataset = dataset.dropna().reset_index(drop=True)
# print(dataset)
# print(dataset.isnull().sum())
print(dataset.iloc[[122]])

# remove the 122nd row
dataset = dataset.drop(122).reset_index(drop=True)
print(dataset.iloc[[122]])

print(dataset.columns)

# fix spaces in columns names
dataset.columns = dataset.columns.str.strip()
# print(dataset.columns)
# print(dataset.info())

# Changes the required columns as integer data type
dataset[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']] = dataset[
    ['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)
# print(dataset.info())

# Changing the other columns to float data datatype
objects = [features for features in dataset.columns if dataset[features].dtypes == 'O']  # Selects All Object types
print(objects)
for i in objects:
    if i != 'Classes':
        dataset[i] = dataset[i].astype(float)
# print(dataset.info())
# print(dataset.describe())
# print(dataset.head())

# Save the cleaned dataset
# dataset.to_csv('Datasets/Algerian_forest_fires_dataset_CLEAN.csv', index=False)

"""Exploratory Data Analysis (EDA)
EDA is a process used to analyze and summarize datasets, identifying characteristics, patterns, and relationships in 
the data before applying machine learning techniques."""
# drop day,month and year
dataset_copy = dataset.drop(['day', 'month', 'year'], axis=1)
# print(dataset_copy.head())

# Encoding of the categories in classes
dataset_copy['Classes'] = np.where(dataset_copy['Classes'].str.contains('not fire'), 0, 1)
# print(dataset_copy.head())

# categories in classes
print(dataset_copy['Classes'].value_counts())
# print(dataset_copy.tail())

# Plot desnity plot for all features
# plt.style.use('seaborn-v0_8')
# dataset_copy.hist(bins=50, figsize=(20, 15))
# plt.show()

# Percentage for Pie Chart
percentage = dataset_copy['Classes'].value_counts(normalize=True) * 100
print(percentage)

# plotting pie chart
# class_labels = ['Fire', 'Not Fire']
# plt.figure(figsize=(12, 7))
# plt.pie(percentage, labels=class_labels, autopct='%1.1f%%')
# plt.title('Pie Chart of Classes')
# plt.show()

# Correlation
print(dataset_copy.corr())

# Heatmap
# sns.heatmap(dataset_copy.corr())
# plt.show()

# Box Plots
# sns.boxplot(dataset_copy['FWI'], color='green')
# plt.show()

# Monthly Fire Analysis
dataset['Classes'] = np.where(dataset['Classes'].str.contains('not fire'), 'not fire', 'fire')

dataset_temp = dataset.loc[dataset['Region'] == 1]
plt.subplots(figsize=(13, 6))
sns.set_style('whitegrid')
sns.countplot(x='month', hue='Classes', data=dataset_temp)
plt.xlabel('Number of Forest Fires', weight='bold')
plt.ylabel('Months', weight='bold')
plt.title('Fire Analysis of Sidi- Bel Region', weight='bold')
plt.show()

dataset_temp = dataset.loc[dataset['Region'] == 0]
plt.subplots(figsize=(13, 6))
sns.set_style('whitegrid')
sns.countplot(x='month', hue='Classes', data=dataset_temp)
plt.xlabel('Number of Forest Fires', weight='bold')
plt.ylabel('Months', weight='bold')
plt.title('Fire Analysis of Bejaia Region', weight='bold')
plt.show()
