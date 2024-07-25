import numpy as np
import matplotlib.pyplot as plt

# Percentiles are used in statistics to give a number that describes the value that a given percent of the values
# are lower than.

ages = [5, 31, 43, 48, 50, 41, 7, 11, 15, 39, 80, 82, 32, 2, 8, 6, 25, 36, 27, 61, 31]
x = np.percentile(ages, 75)
print(x)  # answer is 43, meaning that 75% of the people are 43 or younger.

# What is the age that 90% of the people are younger than?
print(np.percentile(ages, 90))  # answer is 61

ages = sorted(ages)
quantile1, quantile3 = np.percentile(ages, [25, 75])
print(quantile1, quantile3)
print()

vals = np.random.normal(0, 0.5, 10000)
print(np.percentile(vals, 50))
print(np.percentile(vals, 90))
print(np.percentile(vals, 20))

plt.hist(vals, 50)
plt.show()
