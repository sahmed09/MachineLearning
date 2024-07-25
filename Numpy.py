import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""Numpy
NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and 
tools for working with these arrays. It is the fundamental package for scientific computing with Python.

What is an array
An array is a data structure that stores values of same data type. In Python, this is the main difference between 
arrays and lists. While python lists can contain values corresponding to different data types, arrays in python can 
only contain values corresponding to same data type."""

# One dimensional array
my_lst = [1, 2, 3, 4, 5]
arr = np.array(my_lst)
print(arr)
print(type(arr))
print(arr.shape)

# Multinested array
my_lst1 = [1, 2, 3, 4, 5]
my_lst2 = [2, 3, 4, 5, 6]
my_lst3 = [9, 7, 6, 8, 9]

arr = np.array([my_lst1, my_lst2, my_lst3])
print(arr)
print(type(arr))

# check the shape of the array
print(arr.shape)

# Reshape the array
print(arr.reshape(5, 3))

"""Indexing"""

# Accessing the array elements
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr)
print(arr[3])

my_lst1 = [1, 2, 3, 4, 5]
my_lst2 = [2, 3, 4, 5, 6]
my_lst3 = [9, 7, 6, 8, 9]

arr = np.array([my_lst1, my_lst2, my_lst3])
print(arr)
print(arr[1:, 2:])
print(arr[1:, :2])
print(arr[:, :])
print(arr[:, 3:])
print(arr[0:2, 1:4])
print(arr[1, 1:4])

arr = np.arange(0, 10)
print(arr)
arr = np.arange(0, 10, step=2)
print(arr)
arr = np.linspace(0, 10, 20)
print(arr)

# Copy() function and broadcasting
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("arr:", arr)
arr[3:] = [100] * 6
print("arr:", arr)

# Reference type: Sharing the same memory
arr_1 = arr
print("arr_1:", arr_1)
arr_1[3:] = [200] * 6
print("arr_1:", arr_1)
print("arr:", arr)

# To prevent reference type problem, use copy() function
arr_2 = arr.copy()
arr_2[3:] = [800] * 6
print("arr_2:", arr_2)
print("arr:", arr)

# Some conditions very useful in Exploratory Data Analysis
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
val = 2
# print("ass", arr[arr < [8]])

print(np.ones(4))
print(np.ones(4, dtype=int))
print(np.ones((2, 5), dtype=float))

# Create arrays and reshape
arr1 = np.arange(0, 10).reshape(2, 5)
print("arr1:", arr1)
arr2 = np.arange(0, 10).reshape(2, 5)
print("arr2:", arr2)
print(arr1 * arr2)

# Random distribution
print(np.random.rand(3, 3))

arr_ex = np.random.randn(4, 4)  # Return a sample (or samples) from the "standard normal" distribution.
print(arr_ex)

sns.displot(pd.DataFrame(arr_ex.reshape(16, 1)))
# plt.show()

print(np.random.randint(0, 100, 6))
print(np.random.randint(0, 100, 8).reshape(2, 4))
print(np.random.randint(0, 100, 8).reshape(4, 2))
print(np.random.random_sample((1, 5)))
