import pandas as pd
import numpy as np

"""Count or frequency encoding
High Cardinality
Another way to refer to variables that have a multitude of categories, is to call the variables with high cardinality.
If we have categorical variables containing many multiple labels or high cardinality,then by using one hot
encoding, we will expand the feature space dramatically.
One approach that is heavily used in Kaggle competitions, is to replace each label of the categorical variable by
the count, this is the amount of times each label appears in the dataset. Or the frequency, this is the percentage
of observations within that category. The 2 are equivalent."""

data = pd.read_csv('../Datasets/mercedesbenz.csv', usecols=['X1', 'X2'])
print(data.head())
print(data.shape)

# One hot Encoding
print('\nOne hot Encoding')
print(pd.get_dummies(data).shape)
print(len(data['X1'].unique()))
print(len(data['X2'].unique()))

# let's have a look at how many labels
for col in data.columns[0:]:
    print(col, ':', len(data[col].unique()), 'labels')

# Count or frequency encoding
print('\nCount or frequency encoding')

# let's obtain the counts for each one of the labels in variable X2
# let's capture this in a dictionary that we can use to re-map the labels
print(data.X2.value_counts().to_dict())
print(data.head(20))

# And now let's replace each label in X2 by its count
# first we make a dictionary that maps each label to the counts
data_frequency_map = data.X2.value_counts().to_dict()

# and now we replace X2 labels in the dataset data
data.X2 = data.X2.map(data_frequency_map)

print(data.head(20))

"""
Advantages
1. It is very simple to implement
2. Does not increase the feature dimensional space
Disadvantages
1. If some of the labels have the same count, then they will be replaced with the same count and they will loose 
some valuable information.
2. Adds somewhat arbitrary numbers, and therefore weights to the different labels, that may not be related to 
their predictive power
"""
