import matplotlib.pyplot as plt
import seaborn as sns

"""Distribution plots
distplot -> For analyzing dataset with 1 feature
jointplot -> For analyzing dataset with 2 features
pairplot -> For analyzing dataset with more than 2 features
"""

df = sns.load_dataset('tips')
print(df.head())
# Dependent Feature -> tip, Independent Features -> total_bill, sex, smoker, dat, time, size
print(df.dtypes)

"""Correlation with Heatmap
A correlation heatmap uses colored cells, typically in a monochromatic scale, to show a 2D correlation matrix (table) 
between two discrete dimensions or event types. It is very important in Feature Selection.
Correlation can only be found out for integer and flot values (numerical values), not for categorical values.
"""
print(df.corr(numeric_only=True))

# sns.heatmap(df.corr(numeric_only=True))
# plt.show()

"""JoinPlot
A join plot allows to study the relationship between 2 numeric variables. The central chart display their correlation. 
It is usually a scatterplot, a hexbin plot, a 2D histogram or a 2D density plot"""
# Bi-variate Analysis
# sns.jointplot(x='tip', y='total_bill', data=df, kind='hex')
# sns.jointplot(x='tip', y='total_bill', data=df, kind='reg')
# plt.show()

"""Pair plot
A “pairs plot” is also known as a scatterplot, in which one variable in the same data row is matched with another 
variable's value, like this: Pairs plots are just elaborations on this, showing all variables paired with all the 
other variables"""
# sns.pairplot(df)
# sns.pairplot(df, hue='sex')
# plt.show()
print(df['sex'].value_counts())

"""Dist plot
Dist plot helps us to check the distribution of the columns feature"""
# sns.distplot(df['tip'])  # histplot
# plt.show()
# sns.distplot(df['tip'], kde=False, bins=10)
# plt.show()

"""Categorical Plots
Seaborn also helps us in doing the analysis on Categorical Data points.
boxplot
violinplot
countplot
bar plot"""

"""Count plot"""
# sns.countplot(x='day', data=df)
# plt.show()

"""Bar plot"""
# sns.barplot(x='total_bill', y='sex', data=df, hue='sex')
# plt.show()
# sns.barplot(x='sex', y='total_bill', data=df, hue='sex')
# plt.show()

"""Box plot
A box and whisker plot (sometimes called a boxplot) is a graph that presents information from a five-number summary."""
# sns.boxplot(x='smoker', y='total_bill', data=df, hue='smoker')
# plt.show()
# sns.boxplot(x='day', y='total_bill', data=df, palette='rainbow', hue='day')
# plt.show()
# sns.boxplot(data=df, orient='v')
# plt.show()

# categorize my data based on some other categories
# sns.boxplot(x="total_bill", y="day", hue="smoker", data=df)
# plt.show()

"""Violin plot
Violin plot helps us to see both the distribution of data in terms of Kernel density estimation and the box plot"""
# sns.violinplot(x="total_bill", y="day", data=df, palette='rainbow', hue='day')
# plt.show()

iris = sns.load_dataset('iris')
print(iris.head())
print(iris.dtypes)
