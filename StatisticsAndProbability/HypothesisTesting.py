import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest

"""Z Test"""
print('Z Test')
data = [88, 92, 94, 94, 96, 97, 97, 97, 99, 99, 105, 109, 109, 109, 110, 112, 112, 113, 114, 115]

_, p_value = ztest(data, value=110)
print('P-value:', p_value)

if p_value < 0.05:  # alpha value is 0.05 or 5%
    print("we are rejecting null hypothesis, H0")
else:
    print("we are accepting null hypothesis, H0")
print()

"""T Test
A t-test is a type of inferential statistic which is used to determine if there is a significant difference between 
the means of two groups which may be related in certain features"""

ages = [10, 20, 35, 50, 28, 40, 55, 18, 16, 55, 30, 25, 43, 18, 30, 28, 14, 24, 16, 17, 32, 35, 26, 27, 65, 18, 43, 23,
        21, 20, 19, 70]
print('Dataset Size:', len(ages))

ages_mean = np.mean(ages)
print('Ages Mean:', ages_mean)
print()

"""One-sample T-test
The test will tell us whether means of the sample and the population are different"""
print("One-sample T-test")

sample_size = 10
age_sample = np.random.choice(ages, sample_size)
print('Sample Dataset:', age_sample)

ttest, p_value = ttest_1samp(a=age_sample, popmean=30)
print('P-value:', p_value)

if p_value < 0.05:  # alpha value is 0.05 or 5%
    print("we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
print()

np.random.seed(6)
school_ages = stats.poisson.rvs(loc=18, mu=35, size=1500)
classA_ages = stats.poisson.rvs(loc=18, mu=30, size=60)
print('School Ages Mean:', school_ages.mean())
print('Class A Ages Mean:', classA_ages.mean())

_, p_value = stats.ttest_1samp(a=classA_ages, popmean=school_ages.mean())
print('P-value:', p_value)

if p_value < 0.05:  # alpha value is 0.05 or 5%
    print("we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
print()

"""Two-sample T-test
The Independent Samples t Test or 2-sample t-test compares the means of two independent groups in order to determine
whether there is statistical evidence that the associated population means are significantly different."""
print('Two-sample T-test')

np.random.seed(12)
classB_ages = stats.poisson.rvs(loc=18, mu=33, size=60)
print('Class B Ages Mean:', classB_ages.mean())

_, p_value = stats.ttest_ind(a=classA_ages, b=classB_ages, equal_var=False)
print('P-value:', p_value)

if p_value < 0.05:  # alpha value is 0.05 or 5%
    print("we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
print()

"""Paired T-test"""
# When you want to check how different samples from the same group are, you can go for a paired T-test
print('Paired T-test')

weight1 = [25, 30, 28, 35, 28, 34, 26, 29, 30, 26, 28, 32, 31, 30, 45]
weight2 = weight1 + stats.norm.rvs(scale=5, loc=-1.25, size=15)
print('Weight1:', weight1)
print('Weight2:', weight2)

weight_df = pd.DataFrame({'weight_10': np.array(weight1),
                          'weight_20': np.array(weight2),
                          'weight_change': np.array(weight2) - np.array(weight1)})
print(weight_df)

_, p_value = stats.ttest_rel(a=weight1, b=weight2)
print('P-value:', p_value)

if p_value < 0.05:  # alpha value is 0.05 or 5%
    print("we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
print()

"""Correlation"""
print('Correlation')

df = sns.load_dataset('iris')
print(df.shape)
print(df.corr(numeric_only=True))
print()

# sns.pairplot(df)
# plt.show()

"""Chi-Square Test
The test is applied when you have two categorical variables from a single population. It is used to determine whether 
there is a significant association between the two variables."""
print('Chi-Square Test')

dataset = sns.load_dataset('tips')
print(dataset.head())

dataset_table = pd.crosstab(dataset['sex'], dataset['smoker'])
print('Dataset Table: \n', dataset_table)

observed_values = dataset_table.values
print('Observed Values:\n', observed_values)

val = stats.chi2_contingency(dataset_table)
print(val)

expected_values = val[3]
print(expected_values)
print('Degree of Freedom:', val[2])

no_of_rows = len(dataset_table.iloc[0:2, 0])
no_of_columns = len(dataset_table.iloc[0, 0:2])
# row, columns = dataset_table.shape
# print(no_of_rows, no_of_columns)
degree_of_freedom = (no_of_rows - 1) * (no_of_columns - 1)
alpha = 0.05

chi_square = sum(
    [((observed - expected) ** 2) / observed for observed, expected in zip(observed_values, expected_values)])
# print(chi_square)
chi_square_statistic = chi_square[0] + chi_square[1]
print("Chi-Square Statistic:", chi_square_statistic)

critical_value = stats.chi2.ppf(q=1 - alpha, df=degree_of_freedom)
print('Critical Value:', critical_value)

p_value = 1 - stats.chi2.cdf(x=chi_square_statistic, df=degree_of_freedom)
print('P-Value:', p_value)
print('Significance Level:', alpha)
print('Degree of Freedom:', degree_of_freedom)

if chi_square_statistic >= critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")

if p_value <= alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
print()

"""Anova Test
The t-test works well when dealing with two groups, but sometimes we want to compare more than two groups at the 
same time. For example, if we wanted to test whether petal_width age differs based on some categorical variable like 
species, we have to compare the means of each level or group the variable
One Way F-test(Anova)
It tell whether two or more groups are similar or not based on their mean similarity and f-score.
Example : there are 3 different category of iris flowers and their petal width and need to check whether all 3 group 
are similar or not"""
print('Anova Test')

df1 = sns.load_dataset('iris')
print(df1.head())

df_anova = df1[['petal_width', 'species']]
groups = pd.unique(df_anova.species.values)
print('Species Groups:', groups)

d_data = {grp: df_anova['petal_width'][df_anova.species == grp] for grp in groups}
print(d_data)

f, p = stats.f_oneway(d_data['setosa'], d_data['versicolor'], d_data['virginica'])
print('P-Value:', p)

if p < 0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
