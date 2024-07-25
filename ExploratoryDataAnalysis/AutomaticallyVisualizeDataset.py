import seaborn as sns
from ydata_profiling import ProfileReport
import sweetviz as sv
import dtale
from dataprep.datasets import load_dataset
from dataprep.eda import create_report
# from autoviz.AutoViz_Class import AutoViz_Class

# Generating ProfileReport using ydata_profiling (pip install ydata-profiling)
tips = sns.load_dataset('tips')
profile = ProfileReport(tips, explorative=True)
profile.to_file('ydata_profiling_output.html')
# profile.to_widgets()

# A pandas-based library to visualize and compare datasets using sweetviz (pip install sweetviz)
titanic = sns.load_dataset('titanic')
report = sv.analyze(titanic)
report.show_html('titanic_report.html')

# Data Preparation in Python using dataprep
df = load_dataset("titanic")
create_report(df).show_browser()

# Web Client for Visualizing Pandas Objects using dtale (pip install dtale)  (Run in Jupyter Notebook)
# titanic = sns.load_dataset('titanic')
# print(titanic.head())
# print(titanic.shape)
# print(titanic.isnull().sum())
# print(titanic.isna().sum())
#
# dtale.show(titanic)
# d = dtale.show(titanic)

# Automatically Visualize any dataset, any size with a single line of code (pip install autoviz)
# av = AutoViz_Class()
# dft = av.AutoViz('../Datasets/titanic_train.csv', sep=',', depVar='', dfte=None, header=0, verbose=0, lowess=False,
#                  chart_format='svg', max_rows_analyzed=150000, max_cols_analyzed=30)

