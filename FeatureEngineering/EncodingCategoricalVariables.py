import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import category_encoders as ce

"""One Hot Encoding"""
print('One Hot Encoding')
df = pd.DataFrame({'country': ['New York', 'Boston', 'Chicago', 'California', 'New Jersey']})
one_hot = pd.get_dummies(df['country'])
df1 = pd.concat([df, one_hot], axis=1)
df1 = df1.drop('country', axis=1)
print(df1)

df = pd.DataFrame({'country': ['New York', 'Boston', 'Chicago', 'California', 'New Jersey']})
encoder = OneHotEncoder(sparse_output=False)
result = encoder.fit_transform(df)
print(result)

df = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Sex', 'Embarked'])
df.dropna(inplace=True)
one_hot_encoded_data = pd.get_dummies(df, columns=['Sex', 'Embarked'], prefix=['s', 'e'])
print(one_hot_encoded_data.head())

"""Dummy Variable Encoding"""
print('\nDummy Variable Encoding')
data = {'country': ['New York', 'Boston', 'Chicago', 'California', 'New Jersey']}
df = pd.DataFrame(data)
dummy_df = pd.get_dummies(df['country'], drop_first=True, prefix='c')
df = pd.concat([df, dummy_df], axis=1)
print(df)

df = pd.DataFrame({'country': ['New York', 'Boston', 'Chicago', 'California', 'New Jersey']})
encoder = OneHotEncoder(drop='first', sparse_output=False)
result = encoder.fit_transform(df)
print(result)

train = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Sex', 'Embarked'])
Sex = pd.get_dummies(train['Sex'], drop_first=True)
Embarked = pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex', 'Embarked'], axis=1, inplace=True)
train = pd.concat([train, Sex, Embarked], axis=1)
print(train.head())

"""Label Encoding"""
print('\nLabel Encoding')
data = {'country': ['New York', 'Boston', 'Chicago', 'California', 'New Jersey']}
df = pd.DataFrame(data)
label_encoder = LabelEncoder()
df['country_label'] = label_encoder.fit_transform(df['country'])
print(df)

titanic_data = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Sex'])
# print(titanic_data.isnull().sum())
titanic_data['Sex_encoded'] = titanic_data['Sex'].map({'female': 0, 'male': 1})
print(titanic_data.head())

titanic_data = pd.read_csv('../Datasets/titanic_train.csv', usecols=['Sex', 'Embarked'])
titanic_data = titanic_data.fillna(0)
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'].astype(str))
titanic_data['Embarked'] = label_encoder.fit_transform(titanic_data['Embarked'].astype(str))
print(titanic_data.head())

customer_dataset = pd.read_csv('../Datasets/Churn_Modelling.csv', usecols=['Geography', 'Gender'])
label_encoder = LabelEncoder()
customer_dataset['Geography'] = label_encoder.fit_transform(customer_dataset['Geography'])
customer_dataset['Gender'] = label_encoder.fit_transform(customer_dataset['Gender'])
print(customer_dataset.head())

"""Ordinal Encoding"""
print('\nOrdinal Encoding')
df = pd.DataFrame({'quality': ['low', 'medium', 'high', 'medium']})
quality_map = {'low': 0, 'medium': 1, 'high': 2}
df['quality_map'] = df['quality'].map(quality_map)
print(df)

df = pd.DataFrame({'quality': ['low', 'medium', 'high', 'medium']})
encoder = OrdinalEncoder()
result = encoder.fit_transform(df)
print(result)

car_data = pd.read_csv('../Datasets/car_evaluation.csv', header=None)
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
car_data.columns = col_names
print(car_data.head())
# print(car_data.isnull().sum())

X = car_data.drop('class', axis=1)
y = car_data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
cat_encoder = ce.OrdinalEncoder(cols=X_train.columns)
X_train = cat_encoder.fit_transform(X_train)
X_test = cat_encoder.transform(X_test)
print(X_train.head())
print(X_test.head())

"""Binary Encoding"""
print('\nBinary Encoding')
df = pd.DataFrame({'animal': ['cat', 'dog', 'bird', 'cat']})
animal_map = {'cat': 0, 'dog': 1, 'bird': 2}
df['animal'] = df['animal'].map(animal_map)
df['animal'] = df['animal'].apply(lambda x: format(x, 'b'))
print(df)

data = pd.DataFrame({'Country': ['USA', 'Canada', 'UK', 'USA', 'UK']})
encoder = ce.BinaryEncoder(cols=['Country'])
encoded_data = encoder.fit_transform(data)
print(encoded_data)

"""Frequency Encoding or Count Encoding"""
print('\nFrequency Encoding or Count Encoding')
df = pd.DataFrame({'fruit': ['apple', 'banana', 'apple', 'banana', 'apple']})
counts = df['fruit'].value_counts().to_dict()
df['fruit_count'] = df['fruit'].map(counts)
print(df)

"""Target Encoding or Mean Encoding"""
print('\nTarget Encoding or Mean Encoding')
df = pd.DataFrame({'color': ['red', 'green', 'blue', 'red', 'green'], 'target': [0, 1, 0, 1, 0]})
target_mean = df.groupby('color')['target'].mean()
df['color_label'] = df['color'].map(target_mean)
print(df)

data = pd.DataFrame({'color': ['red', 'green', 'blue', 'red', 'green'], 'target': [0, 1, 0, 1, 0]})
encoder = ce.TargetEncoder(cols=['color'])
encoded_data = encoder.fit_transform(df['color'], df['target'])
print(encoded_data)

"""np.where()"""
print('\nnp.where()')
customer_dataset = pd.read_csv('../Datasets/Churn_Modelling.csv', usecols=['Gender'])
print(customer_dataset['Gender'].unique())
customer_dataset['Gender_encode'] = np.where(customer_dataset['Gender'] == 'Male', 1, 0)
print(customer_dataset.tail())
