import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns

# Dealing with Outliers:
# Standard deviation provides a principled way to classify outliers.
# Find data points more than some multiple of a standard deviation in your training data.

dataset = [11, 10, 12, 14, 12, 15, 14, 13, 15, 102, 12, 14, 17, 19, 107, 10, 13, 12, 14, 12, 108, 12, 11, 14, 13, 15,
           10, 15, 12, 10, 14, 13, 15, 10]

"""Outlier detection using z-score"""
# Data point that falls outside of 3 standard deviations. we can use a z score and if the z score falls outside of
# 2 standard deviation
outliers = []


def detect_outliers(data):
    threshold = 3
    mean = np.mean(data)
    std_dev = np.std(data)

    for i in data:
        z_score = (i - mean) / std_dev
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers


outliers_pt = detect_outliers(dataset)
print("Outliers using z-score:", outliers_pt)

"""Outlier detection using interquartile range (IQR)"""
# Data point that falls outside of 1.5 times of an IQR above the 3rd quartile and below the 1st quartile
# Steps
# 1. Arrange the data in increasing order
# 2. Calculate first(q1) and third quartile(q3)
# 3. Find interquartile range (q3-q1)
# 4. Find lower bound q1*1.5
# 5. Find upper bound q3*1.5

dataset = sorted(dataset)
print('Dataset:', dataset)

quantile1, quantile3 = np.percentile(dataset, [25, 75])
print('quantile1:', quantile1, 'quantile3:', quantile3)

iqr_value = quantile3 - quantile1
print('IQR:', iqr_value)

lower_bound_value = quantile1 - (1.5 * iqr_value)
upper_bound_value = quantile3 + (1.5 * iqr_value)
print('Lower Bound:', lower_bound_value, 'Upper Bound:', upper_bound_value)
print("Outliers using IQR:", [value for value in dataset if not lower_bound_value < value < upper_bound_value])
print()


incomes = np.random.normal(27000, 15000, 10000)
incomes = np.append(incomes, [1000000000])

plt.hist(incomes, 50)
plt.title('Income')
plt.show()

print('Before filtering outliers: ')
print("mean =", incomes.mean())
print("median =", np.median(incomes))
print("standard deviation =", np.std(incomes))
print()

"""Outlier Detection Using Standard Deviation"""
# Here's something a little more robust than filtering out billionaires -
# It filters out anything beyond two standard deviations of the median value in the data set:


def reject_outliers_std(data):
    median = np.median(data)
    std_dev = np.std(data)
    lower_limit = median - 2 * std_dev
    upper_limit = median + 2 * std_dev
    filtered = [e for e in data if lower_limit < e < upper_limit]
    return filtered


filtered_value = reject_outliers_std(incomes)
plt.hist(filtered_value, 50)
plt.title('Outlier Removal using Standard Deviation')
plt.show()

print("Outlier Detection Using Standard Deviation")
print("After filtering outliers:")
print("mean =", np.mean(filtered_value))
print("median =", np.median(filtered_value))
print("standard deviation =", np.std(filtered_value))
print('Outliers:', [value for value in incomes if value not in filtered_value])
print()

"""Outlier Detection Using Z-Score"""


def reject_outliers_z(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    filtered = [e for e in data if (-3 < ((e - mean) / std_dev) < 3)]
    return filtered


filtered_z = reject_outliers_z(incomes)
plt.hist(filtered_z, 50)
plt.title('Outlier Removal using z-score')
plt.show()

print("Outlier Detection Using Z-Score")
print("After filtering outliers:")
print("mean =", np.mean(filtered_z))
print("median =", np.median(filtered_z))
print("standard deviation =", np.std(filtered_z))
print('Outliers:', [value for value in incomes if value not in filtered_value])
print()

"""Outlier Detection Using Box Plot"""
plt.boxplot(incomes, vert=True, patch_artist=False)
plt.title('Before Filtering')
plt.show()

filtered_std = reject_outliers_std(incomes)
sns.boxplot(filtered_std)
plt.title('After Filtering')
plt.show()

datasets = sorted(filtered_std)
quantile1, quantile3 = np.percentile(datasets, [25, 75])
iqr_value = quantile3 - quantile1
lower_bound_value = quantile1 - (1.5 * iqr_value)
upper_bound_value = quantile3 + (1.5 * iqr_value)
datasets = [value for value in datasets if lower_bound_value < value < upper_bound_value]

plt.boxplot(datasets, vert=True, patch_artist=False)
plt.title('After Filtering using IQR')
plt.show()

"""Outlier Detection Using DBSCAN"""
np.random.seed(1)
random_data = np.random.randn(50000, 2) * 20 + 20

outlier_detection = DBSCAN(min_samples=2, eps=3)
clusters = outlier_detection.fit_predict(random_data)
print(list(clusters).count(-1))
