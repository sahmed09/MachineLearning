import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# %matplotlib inline

"""Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. 
It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like 
Tkinter, wxPython, Qt, or GTK+.

Some of the major Pros of Matplotlib are:
1. Generally easy to get started for simple plots
2. Support for custom labels and texts
3. Great control of every element in a figure
4. High-quality output in many formats
5. Very customizable in general"""

x = np.arange(-3, 3, 0.001)
# plt.plot(x, norm.pdf(x))
# plt.show()

# Adjust the axes:
axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()

# Add a grid:
# axes = plt.axes()
# axes.set_xlim([-5, 5])
# axes.set_ylim([0, 1.0])
# axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
# axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# axes.grid()
# plt.plot(x, norm.pdf(x))
# plt.plot(x, norm.pdf(x, 1.0, 0.5))
# plt.show()

# Change line types and colors, Labeling Axes and Adding a legend:
# axes = plt.axes()
# axes.set_xlim([-5, 5])
# axes.set_ylim([0, 1.0])
# axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
# axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# axes.grid()
# plt.xlabel('Greebles')
# plt.ylabel('Probability')
# plt.plot(x, norm.pdf(x), 'b-')  # blue solid line
# plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:')  # red dotted line  (r--, r-.)
# plt.legend(['Sneetches', 'Gacks'], loc=1)
# plt.show()

"""Scatter Plot"""
x = np.arange(0, 10)
y = np.arange(11, 21)

# plt.scatter(x, y, c='g')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.title('Graph in 2D')
# plt.show()

X = np.random.randn(500)
Y = np.random.randn(500)
# plt.scatter(X, Y)
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.title('Graph in 2D')
# plt.show()

# Save it to a file:
# points = np.arange(-3, 3, 0.001)
# plt.plot(points, norm.pdf(points))
# plt.plot(points, norm.pdf(points, 1.0, 0.5))
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.title('Graph in 2D')
# plt.savefig('Matplotlib.png', format='png')
# plt.show()

# plt plot
z = x * x
# plt.plot(x, z, 'r*', linestyle='dashed', linewidth=2, markersize=12)
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.title('2D Diagram')
# plt.show()

# Creating Subplots
# plt.subplot(2, 2, 1)
# plt.plot(x, z, 'r--')
# plt.subplot(2, 2, 2)
# plt.plot(x, z, 'g*--')
# plt.subplot(2, 2, 3)
# plt.plot(x, z, 'bo')
# plt.subplot(2, 2, 4)
# plt.plot(x, z, 'go')
# plt.show()

# x = np.arange(1, 11)
# y = 3 * x + 5
# plt.title("Matplotlib demo")
# plt.xlabel("x axis caption")
# plt.ylabel("y axis caption")
# plt.plot(x, y)
# plt.show()

print(np.pi)

# Compute the x and y coordinates for points on a sine curve
# x = np.arange(0, 4 * np.pi, 0.1)
# y = np.sin(x)
# plt.title("sine wave form")
# # Plot the points using matplotlib
# plt.plot(x, y)
# plt.show()

# Subplot()
# Compute the x and y coordinates for points on sine and cosine curves
# x = np.arange(0, 5 * np.pi, 0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)
# # Set up a subplot grid that has height 2 and width 1, and set the first such subplot as active.
# plt.subplot(2, 1, 1)
# # Make the first plot
# plt.plot(x, y_sin, 'r--')
# plt.title('Sine Wave')
# # Set the second subplot as active, and make the second plot.
# plt.subplot(2, 1, 2)
# plt.plot(x, y_cos, 'g--')
# plt.title('Cosine Wave')
# # Show the figure.
# plt.show()

"""Bar plot"""
x1, x2 = [2, 8, 10], [3, 9, 11]
y1, y2 = [11, 16, 9], [6, 15, 7]

# plt.bar(x1, y1, color=['m', 'r', 'y'])
# plt.bar(x2, y2, color=['g', 'b', 'c'])
# # plt.bar(x1, y1)
# # plt.bar(x2, y2, color='g')
# plt.title('Bar graph')
# plt.ylabel('Y axis')
# plt.xlabel('X axis')
# plt.show()

# values = [12, 55, 4, 32, 14]
# colors = ['r', 'g', 'b', 'c', 'm']
# plt.bar(range(0, 5), values, color=colors)
# plt.show()

"""Histograms"""
# a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
# plt.hist(a)
# plt.title("histogram")
# plt.show()

# incomes = np.random.normal(27000, 15000, 10000)
# plt.hist(incomes, 50)
# plt.title("histogram")
# plt.show()

"""Box Plot using Matplotlib"""
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
# rectangular box plot
# plt.boxplot(data, vert=True, patch_artist=False)
# plt.show()
print(data)

"""Pie Chart"""
# Data to plot
values = [215, 130, 245, 210]
labels = ['Python', 'C++', 'Ruby', 'Java']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = [0.1, 0, 0.1, 0]  # explode 1st and 2nd slice

# plt.pie(values, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False)
# plt.title('Programming Language Usage')
# plt.axis('equal')
# plt.show()

"""Box and Whisker Plot"""
# The box represents the two inner quartiles
# dotted line whiskers represent the range of the data except for outliers
uniformSkewed = np.random.rand(100) * 100 - 40
high_outliers = np.random.rand(10) * 50 + 100
low_outliers = np.random.rand(10) * -50 - 100
data = np.concatenate((uniformSkewed, high_outliers, low_outliers))
plt.boxplot(data)
plt.title('Box and Whisker Plot')
plt.show()

# XKCD Style:
# plt.xkcd()
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# plt.xticks([])
# plt.yticks([])
# ax.set_ylim([-30, 10])
#
# data = np.ones(100)
# data[70:] -= np.arange(30)
#
# plt.annotate('The day I realized\nI could cook Bacon\nWhenever I wanted',
#              xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))
# plt.plot(data)
# plt.xlabel('time')
# plt.ylabel('my overall health')
# plt.show()
