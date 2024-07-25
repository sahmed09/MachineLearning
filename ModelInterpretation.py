import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
from lime import lime_tabular

# Model Interpretation using Lime Packages (pip install lime)

dataset = pd.read_csv('Datasets/Churn_Modelling.csv')
print(dataset.head())
print(dataset.isnull().sum())

X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Create dummy variables
geography = pd.get_dummies(X["Geography"], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

# Concatenate the Data Frames
X = pd.concat([X, geography, gender], axis=1)

# Drop Unnecessary columns
X = X.drop(['Geography', 'Gender'], axis=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

pickle.dump(classifier, open("classifier.pkl", 'wb'))

interpretor = lime_tabular.LimeTabularExplainer(training_data=np.array(X_train), feature_names=X_train.columns,
                                                mode='classification')
print(X_test.iloc[4])

exp = interpretor.explain_instance(data_row=X_test.iloc[5], predict_fn=classifier.predict_proba)
exp.save_to_file(file_path='model_interpretation.html', show_table=True)
