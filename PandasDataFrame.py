import pandas as pd
import numpy as np

"""Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data 
analysis tools for the Python programming language."""

# Dataframe
"""A dataframe is a data structure constructed with rows and columns, similar to a database or Excel spreadsheet. 
It consists of a dictionary of lists in which the list each have their own identifiers or keys, such as “last name” 
or “food group. 
A DataFrame is a 2-dimensional data structure designed for handling tabular data with multiple columns, 
while a Series is a 1-dimensional data structure used to represent a single column or row of data within a DataFrame 
or as a standalone data structure.”"""
df = pd.DataFrame(np.arange(0, 20).reshape(5, 4), index=['Row1', 'Row2', 'Row3', 'Row4', 'Row5'],
                  columns=["Column1", "Column2", "Column3", "Column4"])
print(df.head())
# df.to_csv('Datasets/Test')

print(df['Column2'])
print(type(df['Column2']))
print(df[['Column2', 'Column3']])
print(type(df[['Column2', 'Column3']]))

# Accessing the elements 1. .loc, 2. .iloc
print(df.loc['Row1'])

# Check the type
print(type(df.loc['Row1']))

print(df.iloc[:, :])
print(df.iloc[0:2, 0:2])
print(type(df.iloc[0:2, 0:2]))
print(type(df.iloc[0:1, 0:1]))
print(df.iloc[0:2, 0])
print(type(df.iloc[0:2, 0]))
print(type(df.iloc[0, 0:2]))
print(df.iloc[0, 0:2])
print(df.iloc[:, 1:])

# Convert Dataframes into array
print(df.iloc[:, 1:].values)
print(df.iloc[:, 1:].values.shape)

# Check null condition
print(df.isnull().sum())

# Count unique values or categories
print(df['Column1'].value_counts())
print(df['Column1'].unique())
