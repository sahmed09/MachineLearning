import numpy as np
import matplotlib.pyplot as plt
from numpy import dot, mean

"""Correlation:
Just divide the covariance by the standard deviations of both variables, and that normalizes things.
So a correlation of -1 means a perfect inverse correlation
Correlation of 0: no correlation
Correlation 1: perfect correlation"""


def de_mean(x):
    x_mean = mean(x)
    return [xi - x_mean for xi in x]


def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)


def correlation(x, y):
    std_dev_x = x.std()
    std_dev_y = y.std()
    return covariance(x, y) / std_dev_x / std_dev_y


page_speeds = np.random.normal(3.0, 1.0, 1000)  # Mean and standard deviation
purchase_amount = np.random.normal(50.0, 10.0, 1000)
print('Correlation using formula:', correlation(page_speeds, purchase_amount))
print('Correlation using built in function:', np.corrcoef(page_speeds, purchase_amount))
print()

plt.scatter(page_speeds, purchase_amount)
plt.title('Correlation')
plt.xlabel('Page Speeds')
plt.ylabel('Purchase Amounts')
plt.show()

purchase_amount = 100 + page_speeds * 3
print('Positive Correlation')
print('Correlation using formula:', correlation(page_speeds, purchase_amount))
print('Correlation using built in function:', np.corrcoef(page_speeds, purchase_amount))
print()

plt.scatter(page_speeds, purchase_amount)
plt.title('Positive Correlation')
plt.xlabel('Page Speeds')
plt.ylabel('Purchase Amounts')
plt.show()

purchase_amount = 100 - page_speeds * 3
print('Negative Correlation')
print('Correlation using formula:', correlation(page_speeds, purchase_amount))
print('Correlation using built in function:', np.corrcoef(page_speeds, purchase_amount))

plt.scatter(page_speeds, purchase_amount)
plt.title('Negative Correlation')
plt.xlabel('Page Speeds')
plt.ylabel('Purchase Amounts')
plt.show()
