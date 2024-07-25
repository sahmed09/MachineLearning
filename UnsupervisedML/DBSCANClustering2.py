import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

"""Credit Card Dataset"""
print('Credit Card Dataset')
cc_dataset = pd.read_csv('../Datasets/CC_GENERAL.csv')

X = cc_dataset.drop('CUST_ID', axis=1)
X.fillna(method='ffill', inplace=True)  # Handling the missing values
print(X.head())

# Scaling the data to bring all the attributes to a comparable level
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalizing the data so that the data approximately follows a Gaussian distribution
X_normalized = normalize(X_scaled)

# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)

# Reducing the dimensionality of the data to make it visualizable
pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
print(X_principal.head())

# Building the clustering model
db_default = DBSCAN(eps=0.0375, min_samples=3).fit(X_principal)
labels = db_default.labels_

# Visualizing the clustering Building the label to colour mapping
colours = {0: 'r', 1: 'g', 2: 'b', -1: 'k'}

# Building the colour vector for each data point
cvec = [colours[label] for label in labels]

# For the construction of the legend of the plot
r = plt.scatter(X_principal['P1'], X_principal['P2'], color='r')
g = plt.scatter(X_principal['P1'], X_principal['P2'], color='g')
b = plt.scatter(X_principal['P1'], X_principal['P2'], color='b')
k = plt.scatter(X_principal['P1'], X_principal['P2'], color='k')

# Plotting P1 on the X-Axis and P2 on the Y-Axis according to the colour vector defined
plt.figure(figsize=(9, 9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)
plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))
plt.show()

# Tuning the parameters of the model
db = DBSCAN(eps=0.0375, min_samples=50).fit(X_principal)
labels1 = db.labels_

colours1 = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'y', 5: 'm', -1: 'k'}

cvec = [colours1[label] for label in labels]
colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

r = plt.scatter(X_principal['P1'], X_principal['P2'], marker='o', color=colors[0])
g = plt.scatter(X_principal['P1'], X_principal['P2'], marker='o', color=colors[1])
b = plt.scatter(X_principal['P1'], X_principal['P2'], marker='o', color=colors[2])
c = plt.scatter(X_principal['P1'], X_principal['P2'], marker='o', color=colors[3])
y = plt.scatter(X_principal['P1'], X_principal['P2'], marker='o', color=colors[4])
m = plt.scatter(X_principal['P1'], X_principal['P2'], marker='o', color=colors[5])
k = plt.scatter(X_principal['P1'], X_principal['P2'], marker='o', color=colors[6])

plt.figure(figsize=(9, 9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)
plt.legend((r, g, b, c, y, m, k), ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'),
           scatterpoints=1, loc='upper left', ncol=3, fontsize=8)
plt.show()

"""Mall Customers Dataset"""
print('\nMall Customers Dataset')
dataset = pd.read_csv('../Datasets/Mall_Customers.csv')
dataset.rename(columns={'CustomerID': 'customer_id', 'Genre': 'gender', 'Age': 'age', 'Annual Income (k$)': 'income',
                        'Spending Score (1-100)': 'score'}, inplace=True)
print(dataset.head())
print(dataset.info())
print(dataset.describe())

sns.pairplot(dataset)
plt.show()

dataset = dataset.drop(['customer_id'], axis=1)

sns.heatmap(dataset.corr(numeric_only=True))
plt.show()

plt.figure(figsize=(7, 7))
size = dataset['gender'].value_counts()
label = ['Female', 'Male']
color = ['Pink', 'Blue']
explode = [0, 0.1]
plt.pie(size, explode=explode, labels=label, colors=color, shadow=True)
plt.legend()
plt.show()

plt.figure(figsize=(15, 5))
sns.countplot(dataset['income'])
plt.show()

plt.bar(dataset['income'], dataset['score'])
plt.title('Spendscore over income', fontsize=20)
plt.xlabel('Income')
plt.ylabel('Spendscore')
plt.show()

X = dataset.iloc[:, [2, 3]].values
print(X.shape)

cls = DBSCAN(eps=3, min_samples=4, metric='euclidean')
model = cls.fit(X)
label = model.labels_
print(label)

# Identifying the points which makes up our core points
sample_cores = np.zeros_like(label, dtype=bool)
sample_cores[cls.core_sample_indices_] = True

# Calculating the number of clusters
n_clusters = len(set(label)) - (1 if -1 in label else 0)
print('No of clusters:', n_clusters)

y_means = cls.fit_predict(X)
plt.figure(figsize=(7, 5))
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=50, c='pink')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=50, c='yellow')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=50, c='cyan')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s=50, c='magenta')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s=50, c='orange')
plt.scatter(X[y_means == 5, 0], X[y_means == 5, 1], s=50, c='blue')
plt.scatter(X[y_means == 6, 0], X[y_means == 6, 1], s=50, c='red')
plt.scatter(X[y_means == 7, 0], X[y_means == 7, 1], s=50, c='black')
plt.scatter(X[y_means == 8, 0], X[y_means == 8, 1], s=50, c='violet')
plt.xlabel('Annual Income in (1k)')
plt.ylabel('Spending Score from 1-100')
plt.title('Clusters of data')
plt.show()

# HIERARCHICAL CLUSTERING
dendrogram = dendrogram(linkage(X, method='ward'))
plt.title('Dendrogam', fontsize=20)
plt.xlabel('Customers')
plt.ylabel('Ecuclidean Distance')
plt.show()

hc = AgglomerativeClustering(n_clusters=9, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=50, c='pink')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=50, c='yellow')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=50, c='cyan')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=50, c='magenta')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=50, c='orange')
plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s=50, c='blue')
plt.scatter(X[y_hc == 6, 0], X[y_hc == 6, 1], s=50, c='red')
plt.scatter(X[y_hc == 7, 0], X[y_hc == 7, 1], s=50, c='black')
plt.scatter(X[y_hc == 8, 0], X[y_hc == 8, 1], s=50, c='violet')

plt.title('Hierarchial Clustering', fontsize=20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
