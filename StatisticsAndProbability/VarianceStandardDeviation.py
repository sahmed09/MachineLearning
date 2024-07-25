import numpy as np
import math
import matplotlib.pyplot as plt

"""Variance(σ2):
Variance is another number that indicates how spread out the values are or simply the average of the squared 
differences from the mean. In fact, if you take the square root of the variance, you get the standard deviation!
Or the other way around, if you multiply the standard deviation by itself, you get the variance!
Steps:
1. Find the mean -> (32+111+138+28+59+77+97) / 7 = 77.4
2. For each value: find the difference from the mean
    32 - 77.4 = -45.4, 111 - 77.4 = 33.6, 138 - 77.4 = 60.6, 28 - 77.4 = -49.4, 59 - 77.4 = -18.4, 
    77 - 77.4 = -0.4, 97 - 77.4 = 19.6
3. For each difference: find the squared value
    (-45.4)^2 = 2061.16, (33.6)^2 = 1128.96, (60.6)^2 = 3672.36, (-49.4)^2 = 2440.36, (-18.4)^2 = 338.56, 
    (- 0.4)^2 = 0.16, (19.6)^2 = 384.16
4. The variance is the average number of these squared differences
    (2061.16+1128.96+3672.36+2440.36+338.56+0.16+384.16) / 7 = 1432.2"""

speed = [32, 111, 138, 28, 59, 77, 97]
var = np.var(speed)
print("Variance =", var)

"""What is Standard Deviation(σ)?
Standard deviation is a number that describes how spread out the values are.
Standard deviation σ is just the square root of the variance.
A low standard deviation means that most of the numbers are close to the mean (average) value.
A high standard deviation means that the values are spread out over a wider range.
std_dev = sqrt((sum(xi-mean)^2)/N) or sqrt(variance)"""

speed = [32, 111, 138, 28, 59, 77, 97]
var = np.var(speed)
std_dev = np.std(speed)
print("Standard Deviation =", std_dev)
print("Standard Deviation =", math.sqrt(var))
print()

speed = [86, 87, 88, 86, 87, 85, 86]
print("Variance =", np.var(speed))
print("Standard Deviation =", np.std(speed))
print()

incomes = np.random.normal(100.0, 20.0, 10000)
print(incomes.std())
print(incomes.var())
plt.hist(incomes, bins=50)
plt.show()
