from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu
from mlxtend.evaluate import bias_variance_decomp
import warnings

warnings.filterwarnings('ignore')

"""For Classification Problem"""
# Load the dataset
X, y = load_iris(return_X_y=True)

# Split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23, shuffle=True, stratify=y)

# Build the classification model
tree = DecisionTreeClassifier(random_state=123)
clf = BaggingClassifier(estimator=tree, n_estimators=50, random_state=23)

# Bias variance decompositions
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(clf, X_train, y_train, X_test, y_test, loss='0-1_loss',
                                                                 random_seed=23)

# Print the value
print('Average expected loss: %.2f' % avg_expected_loss)
print('Average bias: %.2f' % avg_bias)
print('Average variance: %.2f' % avg_variance)

"""For Regression Problem"""
# Load the dataset
X, y = fetch_california_housing(return_X_y=True)

# Split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23, shuffle=True)

# Build the regression model
model = Sequential([Dense(64, activation=relu), Dense(1)])

# Set optimizer and loss
optimizer = Adam()
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
model.fit(X_train, y_train, epochs=25, verbose=0)
# Evaluations
accuracy = model.evaluate(X_test, y_test)
print('Average: %.2f' % accuracy)

# Bias variance decompositions
avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss='mse',
                                                                 random_seed=23, epochs=5, verbose=0)

# Print the result
print('Average expected loss: %.2f' % avg_expected_loss)
print('Average bias: %.2f' % avg_bias)
print('Average variance: %.2f' % avg_variance)
