import pandas as pd
import datetime

"""Ordinal numbering encoding or Label Encoding
Ordinal categorical variables
Ordinal data is a categorical, statistical data type where the variables have natural, ordered categories
and the distances between the categories is not known.
For example:
Student's grade in an exam (A, B, C or Fail).
Educational level, with the categories: Elementary school, High school, College graduate, PhD ranked from 1 to 4.
When the categorical variables are ordinal, the most straightforward best approach is to replace the labels by
some ordinal number based on the ranks."""

# create a variable with dates, and from that extract the weekday
# I create a list of dates with 20 days difference from today and then transform it into a datafame
df_base = datetime.datetime.today()
df_date_list = [df_base - datetime.timedelta(days=x) for x in range(0, 20)]
df = pd.DataFrame(df_date_list)
df.columns = ['day']
print(df)

# extract the week day name
df['day_of_week'] = df['day'].dt.day_name()
print(df.head())

# Engineer categorical variable by ordinal number replacement
weekday_map = {'Monday': 1,
               'Tuesday': 2,
               'Wednesday': 3,
               'Thursday': 4,
               'Friday': 5,
               'Saturday': 6,
               'Sunday': 7}
df['day_ordinal'] = df.day_of_week.map(weekday_map)
print(df.head())

"""
Ordinal Measurement Advantages
Ordinal measurement is normally used for surveys and questionnaires. Statistical analysis is applied to the responses 
once they are collected to place the people who took the survey into the various categories. The data is then compared 
to draw inferences and conclusions about the whole surveyed population with regard to the specific variables. The 
advantage of using ordinal measurement is ease of collation and categorization. If you ask a survey question without 
providing the variables, the answers are likely to be so diverse they cannot be converted to statistics.
With Respect to Machine Learning
1. Keeps the semantical information of the variable (human readable content)
2. Straightforward

Ordinal Measurement Disadvantages
The same characteristics of ordinal measurement that create its advantages also create certain disadvantages. The 
responses are often so narrow in relation to the question that they create or magnify bias that is not factored into 
the survey. For example, on the question about satisfaction with the governor, people might be satisfied with his job 
performance but upset about a recent sex scandal. The survey question might lead respondents to state their 
dissatisfaction about the scandal, in spite of satisfaction with his job performance -- but the statistical 
conclusion will not differentiate.
With Respect to Machine Learning
1. Does not add machine learning valuable information
"""
