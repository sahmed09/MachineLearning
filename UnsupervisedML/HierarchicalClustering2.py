import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# We will use Agglomerative Clustering, a type of hierarchical clustering that follows a bottom up approach. We begin
# by treating each data point as its own cluster. Then, we join clusters together that have the shortest distance
# between them to create larger clusters. This step is repeated until one large cluster is formed containing all of
# the data points.

# compute the ward linkage using euclidean distance, and visualize it using a dendrogram
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
plt.show()

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
Z = linkage(X, 'ward')
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
plt.show()

# AgglomerativeClustering with Python's scikit-learn library
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

hierarchical_cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)
print('Labels (AgglomerativeClustering):', labels)
plt.scatter(x, y, c=labels)
plt.show()

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
# here we need to mention the number of clusters, otherwise the result will be a single cluster containing all the data
clustering = AgglomerativeClustering(n_clusters=2).fit(X)
print(clustering.labels_)

"""Mall Customers Dataset"""
print('\nMall Customers Dataset')
dataset = pd.read_csv('../Datasets/Mall_Customers.csv')
print(dataset.head())

X = dataset.iloc[:, [3, 4]].values

# Finding the optimal number of clusters using the dendrogram
dendro = dendrogram(linkage(X, method='ward'))
plt.title("Dendrogrma Plot")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()

# Training the hierarchical model on dataset
h_clustering = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_pred = h_clustering.fit_predict(X)

# visulaizing the clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=100, c='blue', label='Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=100, c='green', label='Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=100, c='red', label='Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

"""IRIS Dataset"""
print('\nIRIS Dataset')
data_path = sns.load_dataset('iris')
print(data_path.head())
print(data_path['species'].unique(), data_path['species'].nunique())
X = data_path[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
print(X.head())

model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
y_pred = model.fit_predict(X)
print(y_pred)

plt.figure(figsize=(15, 6))
link_matrix = linkage(X, method='ward')
dendrogram = dendrogram(link_matrix)
plt.title('Dendrogram')
plt.xlabel('Flowers')
plt.ylabel('Euclidean distances')
plt.show()
