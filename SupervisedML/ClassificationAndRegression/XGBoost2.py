import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

diamonds_dataset = sns.load_dataset('diamonds')
print(diamonds_dataset.head())
print(diamonds_dataset.shape)
print(diamonds_dataset.describe())
print(diamonds_dataset.info())
print(diamonds_dataset.describe(exclude=np.number))

# Build an XGBoost DMatrix
X, y = diamonds_dataset.drop('price', axis=1), diamonds_dataset['price']

# Extract text features
# cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
# for col in cats:
#     X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {'objective': 'reg:squarederror', 'tree_method': 'gpu_hist'}
model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=100)
preds = model.predict(dtest_reg)

rmse = mean_squared_error(y_test, preds, squared=False)
print(f'RMSE of the base model: {rmse:.3f}')

# params = {'objective': 'reg:squarederror', 'tree_method': 'gpu_hist'}
# evals = [(dtrain_reg, 'train'), (dtest_reg, 'validation')]
# model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=100, evals=evals)

# params = {'objective': 'reg:squarederror', 'tree_method': 'gpu_hist'}
# evals = [(dtest_reg, "validation"), (dtrain_reg, "train")]
# model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=100, evals=evals, verbose_eval=10)

# params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
# evals = [(dtest_reg, "validation"), (dtrain_reg, "train")]
# # model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=5000, evals=evals, verbose_eval=500)
# model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=10000, evals=evals, verbose_eval=50,
#                   early_stopping_rounds=50)

# XGBoost Cross-Validation
params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
results = xgb.cv(params, dtrain_reg, num_boost_round=1000, nfold=5, early_stopping_rounds=20)
print(results.head())

best_rmse = results['test-rmse-mean'].min()
print(best_rmse)

"""XGBoost Classification"""
print('\nXGBoost Classification')
X, y = diamonds_dataset.drop("cut", axis=1), diamonds_dataset[['cut']]

# Encode y to numeric
y_encoded = OrdinalEncoder().fit_transform(y)

# Extract text features
# cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to pd.Categorical
# for col in cats:
#     X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=1, stratify=y_encoded)

# Create classification matrices
dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_clf = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {"objective": "multi:softprob", "tree_method": "gpu_hist", "num_class": 5}
results = xgb.cv(params, dtrain_clf, num_boost_round=1000, nfold=5, metrics=["mlogloss", "auc", "merror"])
print(results.keys())
print(results['test-auc-mean'].max())

# XGBoost Native vs. XGBoost Sklearn
xgb_classifier = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', tree_method='hist', eta=0.1,
                                   max_depth=3, enable_categorical=True)
xgb_classifier.fit(X_train, y_train)

# Convert the model to a native API model
model = xgb_classifier.get_booster()
