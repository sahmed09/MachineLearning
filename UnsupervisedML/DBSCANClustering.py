import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

"""Mall Customers Dataset"""
print('Mall Customers Dataset')
dataset = pd.read_csv('../Datasets/Mall_Customers.csv')
print(dataset.head())

X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
dbscan = DBSCAN(eps=3, min_samples=4)
model = dbscan.fit(X)

labels = model.labels_
print(labels)

# identifying the points which makes up our core points
sample_cores = np.zeros_like(labels, dtype=bool)  # Set everything as False
sample_cores[dbscan.core_sample_indices_] = True

# Calculating the number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print('No of Clusters:', n_clusters)

print('Silhouette Score:', silhouette_score(X, labels))

"""Mall Customers Dataset"""
print('\nMall Customers Dataset')
dataset = pd.read_csv('../Datasets/Mall_Customers.csv')
dataset.rename(columns={'CustomerID': 'customer_id', 'Genre': 'gender', 'Age': 'age', 'Annual Income (k$)': 'income',
                        'Spending Score (1-100)': 'score'}, inplace=True)
print(dataset.head())
print(dataset.info())
print(dataset.describe())

dataset = dataset.drop(['customer_id'], axis=1)

features = ['age', 'income', 'score']
X_train = dataset[features]

cls = DBSCAN(eps=12.5, min_samples=4).fit(X_train)
datasetDBSCAN = X_train.copy()
datasetDBSCAN.loc[:, 'cluster'] = cls.labels_
datasetDBSCAN.cluster.value_counts().to_frame()

outliers = datasetDBSCAN[datasetDBSCAN['cluster'] == -1]

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
sns.scatterplot(x='income', y='score', data=datasetDBSCAN[datasetDBSCAN['cluster'] != -1], hue='cluster', ax=ax[0],
                palette='Set3', legend='full', s=180)
sns.scatterplot(x='age', y='score', data=datasetDBSCAN[datasetDBSCAN['cluster'] != -1], hue='cluster', palette='Set3',
                ax=ax[1], legend='full', s=180)
ax[0].scatter(outliers['income'], outliers['score'], s=9, label='outliers', c="k")
ax[1].scatter(outliers['age'], outliers['score'], s=9, label='outliers', c="k")
ax[0].legend()
ax[1].legend()
plt.setp(ax[0].get_legend().get_texts(), fontsize='11')
plt.setp(ax[1].get_legend().get_texts(), fontsize='11')
plt.show()

"""Make Blobs"""
print('\nMake Blobs')
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
# X, labels_true = make_blobs(n_samples=750, centers=3, cluster_std=0.4, random_state=0)

X = StandardScaler().fit_transform(X)

# Compute DBSCAN
db_scan = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db_scan.labels_, dtype=bool)
core_samples_mask[db_scan.core_sample_indices_] = True
labels = db_scan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity Score: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness Score: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure Score: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

# Plot result
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

"""Make Circles"""
print('\nMake Circles')
X, y = make_circles(n_samples=750, factor=0.3, noise=0.1)
X = StandardScaler().fit_transform(X)
y_pred = DBSCAN(eps=0.3, min_samples=10).fit_predict(X)

print('Number of clusters: {}'.format(len(set(y_pred[np.where(y_pred != -1)]))))
print('Homogeneity: {}'.format(metrics.homogeneity_score(y, y_pred)))
print('Completeness: {}'.format(metrics.completeness_score(y, y_pred)))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
