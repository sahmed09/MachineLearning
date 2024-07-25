import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

"""Moments: Quantitative measures of the shape of a probability density function.
The first moment is the mean.
The second moment is the variance.
The third moment is the skew.
    A distribution with a longer tail on the left will be skewed left, and have a negative skew.
    A distribution with a longer tail on the right will be skewed right, and have a positive skew.
The fourth moment is the kurtosis.
    Kurtosis tells you how a data distribution compares in shape to a normal (Gaussian) distribution"""

vals = np.random.normal(0, 0.5, 10000)
mean = np.mean(vals)  # first moment
variance = np.var(vals)  # second moment
skewness = sp.skew(vals)  # third moment
kurtosis = sp.kurtosis(vals)  # fourth moment

print('Mean: {} \nVariance: {} \nSkewness: {} \nKurtosis: {}'.format(mean, variance, skewness, kurtosis))

plt.hist(vals, 50)
plt.show()
