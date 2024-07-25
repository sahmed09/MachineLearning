import pandas as pd
import numpy as np

"""Read Json to CSV"""

Data = '{"employee_name": "James", "email": "james@gmail.com", "job_profile": [{"title1":"Team Lead", "title2":"Sr. Developer"}]}'
df = pd.read_json(Data)
print(df)

# convert Json to different json formats
df1 = df.to_json()  # without orient, it will create key:value pair for every columns
print(df1)
df1 = df.to_json(orient='records')
print(df1)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
print(df.head())

# convert Json to csv
# df.to_csv('Datasets/wine.csv')

# convert Json to different json formats
wine = df.to_json(orient='records')
print(wine)
print(df.to_json(orient='index'))

"""Reading HTML content"""
url = 'https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/'
df = pd.read_html(url)
print(type(df))
print(df[0])

url_mcc = 'https://en.wikipedia.org/wiki/Mobile_country_code'
df = pd.read_html(url_mcc, match='Country', header=0)
print(df[0])

"""Reading Excel Files"""
df_excel = pd.read_excel('Datasets/Excel_Sample.xlsx')
print(df_excel.head())

"""Pickling
All pandas objects are equipped with to_pickle methods which use Python’s cPickle module to save data structures 
to disk using the pickle format."""
df_excel.to_pickle('Datasets/df_excel')
df = pd.read_pickle('Datasets/df_excel')
print(df.head())
