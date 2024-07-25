import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)

predicted = logistic_regression.predict(np.array([3.49]). reshape(-1, 1))
print('New Prediction:', predicted)
predicted = logistic_regression.predict(np.array([3.46]). reshape(-1, 1))
print('New Prediction:', predicted)

log_odds = logistic_regression.coef_
odds = np.exp(log_odds)
print(log_odds)


def logit2prob(logr, x):
    log_odds = logr.coef_ * x + logr.intercept_
    odds = np.exp(log_odds)
    probability = odds / (1 + odds)
    return probability


print(logit2prob(logistic_regression, X))
