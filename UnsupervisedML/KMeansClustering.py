import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings

warnings.filterwarnings('ignore')

# Choosing the K value
print('Elbow method')
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, n_init=10)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

print('\nSilhouette method')
sil_avg = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    sil_avg.append(silhouette_score(X, labels, metric='euclidean'))

plt.plot(range_n_clusters, sil_avg, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')
plt.show()

X = -2 * np.random.rand(100, 2)
X1 = 1 + 2 * np.random.rand(50, 2)
X[50:100, :] = X1
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.show()

kmean = KMeans(n_clusters=2, n_init=10)
kmean.fit(X)

cluster_centers = kmean.cluster_centers_
print('Center of the clusters (Centroid):', cluster_centers)

plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.scatter(cluster_centers[0][0], cluster_centers[0][1], s=200, c='g', marker='s')
plt.scatter(cluster_centers[1][0], cluster_centers[1][1], s=200, c='r', marker='s')
plt.show()

print(kmean.labels_)

sample_test = np.array([-3.0, -3.0])
second_test = sample_test.reshape(1, -1)
print(kmean.predict(second_test))

"""California Housing Dataset (Default)"""
print('\nCalifornia Housing Dataset (Default)')
dataset = fetch_california_housing(as_frame=True)
X = dataset.data
print(X.head())
y = dataset.target

kmeans = KMeans(n_clusters=6)
X['Cluster'] = kmeans.fit_predict(X)
X['Cluster'] = X['Cluster'].astype('category')
print(X.head())

sns.relplot(x='Longitude', y='Latitude', hue='Cluster', data=X, height=6)
plt.show()

# X["MedHouseVal"] = dataset["MedHouseVal"]
sns.catplot(x=y, y="Cluster", data=X, kind="boxen", height=6)
plt.show()

"""California Housing Dataset (CSV)"""
print('\nCalifornia Housing Dataset (CSV)')
home_data = pd.read_csv('../Datasets/california_housing.csv', usecols=['longitude', 'latitude', 'median_house_value'])
print(home_data.head())

sns.scatterplot(data=home_data, x='longitude', y='latitude', hue='median_house_value')
plt.show()

# Normalizing the Data
X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']],
                                                    home_data[['median_house_value']], test_size=0.33, random_state=0)
X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)

kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(X_train_norm)

sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=kmeans.labels_)
plt.show()

sns.boxplot(x=kmeans.labels_, y=y_train['median_house_value'])
plt.show()

sil_score = silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')
print('Silhouette Score:', sil_score)

# Choosing the best number of clusters
K = range(2, 8)
wcss_list = []  # SUM OF SQUARED ERROR
silh_score = []

for k in K:
    model = KMeans(n_clusters=k, random_state=0, n_init='auto')
    model.fit(X_train_norm)
    wcss_list.append(model)
    # Append the silhouette score to scores
    silh_score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=wcss_list[0].labels_)
plt.show()

sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=wcss_list[2].labels_)
plt.show()

sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=wcss_list[2].labels_)
plt.show()

sns.lineplot(x=K, y=silh_score)
plt.show()

sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=wcss_list[3].labels_)
plt.show()

sns.boxplot(x=wcss_list[3].labels_, y=y_train['median_house_value'])
plt.show()

"""IRIS Dataset"""
print('\nIRIS Dataset')
X, y = load_iris(return_X_y=True)

# Find optimum number of cluster
sse = []  # SUM OF SQUARED ERROR
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=2, n_init=10)
    km.fit(X)
    sse.append(km.inertia_)

sns.set_style("whitegrid")
g = sns.lineplot(x=range(1, 11), y=sse)
g.set(xlabel="Number of cluster (k)", ylabel="Sum Squared Error", title='Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=2, n_init=10)
kmeans.fit(X)
print(kmeans.cluster_centers_)

pred = kmeans.predict(X)
print(pred)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
    center = center[:2]
    plt.scatter(center[0], center[1], marker='^', c='red')
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 2], X[:, 3], c=pred, cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
    center = center[2:4]
    plt.scatter(center[0], center[1], marker='^', c='red')
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.show()

customer_data = pd.read_csv('../Datasets/Mall_Customers.csv')
print(customer_data.head())
print(customer_data.isna().sum())  # check for null or missing values

plt.scatter(customer_data['Annual Income (k$)'], customer_data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Initialize centroids randomly
K = 3
centroids = customer_data.sample(n=K)
plt.scatter(customer_data['Annual Income (k$)'], customer_data['Spending Score (1-100)'])
plt.scatter(centroids['Annual Income (k$)'], centroids['Spending Score (1-100)'], c='black')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Implementation K-means for K=3
km_sample = KMeans(n_clusters=3, n_init='auto')
km_sample.fit(customer_data[['Annual Income (k$)', 'Spending Score (1-100)']])

labels_sample = km_sample.labels_
customer_data['label'] = labels_sample

# Label 0: Savers, avg to high income but spend wisely, Label 1: Carefree, low income but spenders,
# Label 2: Spenders, avg to high income and spender
sns.scatterplot(data=customer_data, x='Annual Income (k$)', y='Spending Score (1-100)', hue=customer_data['label'],
                palette='Set1')
plt.show()

X = customer_data.iloc[:, [3, 4]].values

# finding optimal number of clusters using the elbow method
wcss_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss_list.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
y_predict = kmeans.fit_predict(X)

plt.scatter(X[y_predict == 0, 0], X[y_predict == 0, 1], s=100, c='blue', label='Cluster 1')  # for first cluster
plt.scatter(X[y_predict == 1, 0], X[y_predict == 1, 1], s=100, c='green', label='Cluster 2')  # for second cluster
plt.scatter(X[y_predict == 2, 0], X[y_predict == 2, 1], s=100, c='red', label='Cluster 3')  # for third cluster
plt.scatter(X[y_predict == 3, 0], X[y_predict == 3, 1], s=100, c='cyan', label='Cluster 4')  # for fourth cluster
plt.scatter(X[y_predict == 4, 0], X[y_predict == 4, 1], s=100, c='magenta', label='Cluster 5')  # for fifth cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroid')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
