import numpy as np
import matplotlib.pyplot as plt
from numpy import dot, mean

"""
Measuring covariance:
Think of the data sets for the two variables as high-dimensional vectors.
Convert these to vectors of variances from the mean.
Take the dot product (cosine of the angle between them) of the two vectors.
Divide by the sample size.
"""


def de_mean(x):
    x_mean = mean(x)
    return [xi - x_mean for xi in x]


def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)


page_speeds = np.random.normal(3.0, 1.0, 1000)
purchase_amount = np.random.normal(50.0, 10.0, 1000)
print('Covariance using formula:', covariance(page_speeds, purchase_amount))
print('Covariance using built in function:', np.cov(page_speeds, purchase_amount))
print()

plt.scatter(page_speeds, purchase_amount)
plt.title('Covariance')
plt.xlabel('Page Speeds')
plt.ylabel('Purchase Amounts')
plt.show()

purchase_amount = np.random.normal(50.0, 10.0, 1000) / page_speeds
print('Covariance using formula:', covariance(page_speeds, purchase_amount))
print('Covariance using built in function:', np.cov(page_speeds, purchase_amount))
print()

plt.scatter(page_speeds, purchase_amount)
plt.title('Covariance')
plt.xlabel('Page Speeds')
plt.ylabel('Purchase Amounts')
plt.show()

purchase_amount = 100 + page_speeds * 3
print('Positive Covariance')
print('Covariance using formula:', covariance(page_speeds, purchase_amount))
print('Covariance using built in function:', np.cov(page_speeds, purchase_amount))
print()

plt.scatter(page_speeds, purchase_amount)
plt.title('Positive Covariance')
plt.xlabel('Page Speeds')
plt.ylabel('Purchase Amounts')
plt.show()

purchase_amount = 100 - page_speeds * 3
print('Negative Covariance')
print('Covariance using formula:', covariance(page_speeds, purchase_amount))
print('Covariance using built in function:', np.cov(page_speeds, purchase_amount))

plt.scatter(page_speeds, purchase_amount)
plt.title('Negative Covariance')
plt.xlabel('Page Speeds')
plt.ylabel('Purchase Amounts')
plt.show()
