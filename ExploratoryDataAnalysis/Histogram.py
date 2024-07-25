import numpy as np
import matplotlib.pyplot as plt

a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
plt.hist(a)
plt.title("histogram")
plt.show()

incomes = np.random.normal(27000, 15000, 10000)
plt.hist(incomes, 50)
plt.title("histogram")
plt.show()
