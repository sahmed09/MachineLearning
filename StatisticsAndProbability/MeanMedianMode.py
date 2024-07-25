import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
Mean - The average value
Median - The mid point value
Mode - The most common value
"""

"""Mean:
The mean value is the average value.
To calculate the mean, find the sum of all values, and divide the sum by the number of values."""
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
x = np.mean(speed)
print("Mean =", x)

"""Median:
The median value is the value in the middle, after you have sorted all the values.
It is important that the numbers are sorted before you can find the median."""
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
x = np.median(speed)
print("Median =", x)

# If there are two numbers in the middle, divide the sum of those numbers by two.
speed = [99, 86, 87, 88, 86, 103, 87, 94, 78, 77, 85, 86]
x = np.median(speed)
print("Median =", x)

"""Mode:
The Mode value is the value that appears the most number of times.
The SciPy module has a method for this."""
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
x = stats.mode(speed, keepdims=False)
print("Mode =", x)
print()

incomes = np.random.normal(100.0, 20.0, 10000)
print('Mean:', np.mean(incomes))
print('Median:', np.median(incomes))
plt.hist(incomes, bins=50)
plt.show()
