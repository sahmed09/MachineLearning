import evalml
import pandas as pd
from evalml.automl import AutoMLSearch

# EvalML is an AutoML library that builds, optimizes, and evaluates machine learning pipelines using domain-specific
# objective functions. (pip install evalml)

X, y = evalml.demos.load_breast_cancer()
X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='binary')
print(X_train.head())

# Running the AutoML to select the best algorithm
algorithms = evalml.problem_types.ProblemTypes.all_problem_types
print(algorithms)

automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary')
automl.search()
ranks = automl.rankings
print(ranks)
print(ranks.columns)
print(ranks[['id', 'pipeline_name', 'mean_cv_score']])

# Getting The Best Pipeline
best_pipeline = automl.best_pipeline
print(best_pipeline)

# Let's Check the detailed description
print(automl.describe_pipeline(ranks.iloc[0]['id']))

# Evaluate on hold out data
best_pipeline_score = best_pipeline.score(X_test, y_test, objectives=['auc', 'f1', 'precision', 'recall'])
print(best_pipeline_score)

# We can also optimize for a problem specific objective
automl_auc = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary', objective='auc',
                          additional_objectives=['f1', 'precision'], max_batches=1, optimize_thresholds=True)
automl_auc.search()
ranks_auc = automl_auc.rankings
print(ranks_auc)
print(ranks_auc[['id', 'pipeline_name', 'mean_cv_score']])
print(automl_auc.describe_pipeline(ranks_auc.iloc[0]['id']))

best_pipeline_auc = automl_auc.best_pipeline

# get the score on holdout data
best_pipeline_auc_score = best_pipeline_auc.score(X_test, y_test,  objectives=["auc"])
print(best_pipeline_auc_score)

best_pipeline.save("model.pkl")

# Loading the Model
check_model = automl.load('model.pkl')

predictions = check_model.predict_proba(X_test)
print(predictions)

"""Titanic Dataset"""
print('\nTitanic Dataset')
df = pd.read_csv('../Datasets/titanic_train.csv')
X = df.iloc[:, 2:]
X = X.drop('Name', axis=1)
y = df.iloc[:, 1]

X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='binary')

automl_titanic = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary')
automl_titanic.search()
ranks = automl_titanic.rankings
print(ranks)
print(ranks[['id', 'pipeline_name', 'mean_cv_score']])

best_pipeline = automl_titanic.best_pipeline
print(best_pipeline)

# Let's Check the detailed description
print(automl_titanic.describe_pipeline(ranks.iloc[0]['id']))

# Evaluate on hold out data
best_pipeline_score = best_pipeline.score(X_test, y_test, objectives=['auc', 'f1', 'precision', 'recall'])
print(best_pipeline_score)
