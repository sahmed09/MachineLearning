import pandas as pd
import seaborn as sns
import graphviz
import matplotlib.pyplot as plt
import category_encoders as ce
from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

"""Car Dataset"""
car_data = pd.read_csv('../../Datasets/car_evaluation.csv', header=None)

# 1. Exploratory Data Analysis
print(car_data.head())
print(car_data.shape)

# Rename column names
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
car_data.columns = col_names
print(car_data.head())
print(car_data.info())

# Frequency distribution of values in variables
for col in col_names:
    print(car_data[col].value_counts())

# Check missing values in variables
print(car_data.isnull().sum())

# 2. Declare feature vector and target variable
X = car_data.drop('class', axis=1)
y = car_data['class']

# 3. Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)

# 4. Feature Engineering
# check data types in X_train
print(X_train.dtypes)
print(X_train.head())

# Encode categorical variables
# cat_encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
cat_encoder = ce.OrdinalEncoder(cols=X_train.columns)
X_train = cat_encoder.fit_transform(X_train)
X_test = cat_encoder.transform(X_test)
print(X_train.head())
print(X_test.head())

# 5. Random Forest Classifier model with default parameters
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X_train, y_train)
y_predictions = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
print('Model accuracy score with 10 decision-trees : {0:0.4f}'.format(accuracy))

# 6. Random Forest Classifier model with parameter n_estimators=100
rf_classifier_100 = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier_100.fit(X_train, y_train)
y_predictions_100 = rf_classifier_100.predict(X_test)

accuracy_100 = accuracy_score(y_test, y_predictions)
print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(accuracy_100))

# 7. Find important features with Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feature_scores)

# 8. Visualize feature scores of the features
sns.barplot(x=feature_scores, y=feature_scores.index, hue=feature_scores)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# 9. Build Random Forest model on selected features
X = car_data.drop(['class', 'doors'], axis=1)
y = car_data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.head())

# cat_encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])
cat_encoder = ce.OrdinalEncoder(cols=X_train.columns)
X_train = cat_encoder.fit_transform(X_train)
X_test = cat_encoder.transform(X_test)

rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X_train, y_train)
y_predictions = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
precision = precision_score(y_test, y_predictions, average='weighted')
recall = recall_score(y_test, y_predictions, average='weighted')
f1 = f1_score(y_test, y_predictions, average='weighted')
print('Model accuracy score with doors variable removed : {0:0.4f}'.format(accuracy))
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

c_matrix = confusion_matrix(y_test, y_predictions)
print('Confusion matrix\n\n', c_matrix)

report = classification_report(y_test, y_predictions)
print(report)

"""Bank Marketing Dataset"""
bank_data = pd.read_csv('../../Datasets/bank-marketing.csv', sep=';')
bank_data = bank_data.loc[:, ['age', 'default', 'cons.price.idx', 'cons.conf.idx', 'y']]
print(bank_data.head())

# Preprocessing Data for Random Forests
bank_data['default'] = bank_data['default'].map({'no': 0, 'yes': 1, 'unknown': 0})
bank_data['y'] = bank_data['y'].map({'no': 0, 'yes': 1})

# Split the data into features (X) and target (y)
X = bank_data.drop('y', axis=1)
y = bank_data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Export the first three decision trees from the forest (Visualizing the Results)
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree, feature_names=X_train.columns, filled=True, max_depth=2, impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    graph.view()

# Hyperparameter Tuning
param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

print('Best hyperparameters:', rand_search.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)

ConfusionMatrixDisplay(confusion_matrix=c_matrix).plot()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Create a series contain feature importance from the model and feature names from the training data
feature_importance = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feature_importance)

# Plot a simple bar chart
feature_importance.plot.bar()
plt.show()

sns.barplot(x=feature_importance, y=feature_importance.index, hue=feature_importance)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()
