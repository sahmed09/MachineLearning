import pandas as pd
import numpy as np

df = pd.read_csv('Datasets/mercedesbenz.csv')
print(df.head())
print(df.info())
print(df.describe())  # It only takes integer and flot columns into consideration, skips categorical features

# Get the unique category counts
print(df['X0'].value_counts())

print(df[df['y'] > 100])
