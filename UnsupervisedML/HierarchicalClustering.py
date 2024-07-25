import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_wine

"""Wine Dataset"""
print('Wine Dataset')
data = load_wine()
X = data.data

# Convert to DataFrame
wine_data = pd.DataFrame(X, columns=data.feature_names)
print(wine_data.head())
print(wine_data.shape)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform Hierarchical Clustering and Plot the Dendrogram
linked_matrix = linkage(X_scaled, method='ward')

plt.figure(figsize=(9, 5), dpi=200)
dendrogram(linked_matrix, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Truncating the Dendrogram for Easier Visualization
linked_matrix = linkage(X_scaled, method='ward')

plt.figure(figsize=(7, 4), dpi=200)
dendrogram(linked_matrix, orientation='top', distance_sort='descending', truncate_mode='level', p=3,
           show_leaf_counts=True)
plt.title('Truncated Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Identify the Optimal Number of Clusters (optimal number of clusters is 3.)
# Form the Clusters
# Choose a threshold distance based on the dendrogram
threshold_distance = 3.5

# Cut the dendrogram to get cluster labels
cluster_labels = fcluster(linked_matrix, threshold_distance, criterion='distance')

# Assign cluster labels to the DataFrame
wine_data['cluster'] = cluster_labels
print(wine_data['cluster'])

# Visualize the Clusters
plt.figure(figsize=(8, 5))
scatter = plt.scatter(wine_data['alcohol'], wine_data['flavanoids'], c=wine_data['cluster'], cmap='rainbow')
plt.xlabel('Alcohol')
plt.ylabel('Flavonoids')
plt.title('Visualizing the clusters')

legend_labels = [f'Cluster {i + 1}' for i in range(3)]  # n_clusters=3
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
plt.show()

"""Loan Dataset"""
print('\nLoan Dataset')
loan_data = pd.read_csv('../Datasets/loan_data.csv')
print(loan_data.shape)
print(loan_data.head())
print(loan_data.info())
print(loan_data['not.fully.paid'].value_counts())

# Preprocessing the data
# Deal with missing values (compute percentage in each columns)
percent_missing = round(100 * (loan_data.isnull().sum()) / len(loan_data), 2)
print(percent_missing)

# Drop unwanted columns
cleaned_data = loan_data.drop(['purpose', 'not.fully.paid'], axis=1)
print(cleaned_data.head())


# Outliers analysis
def show_boxplot(df):
    plt.rcParams['figure.figsize'] = [14, 6]
    sns.boxplot(data=df, orient='v')
    plt.title('Outliers Distribution', fontsize=16)
    plt.xlabel('Attributes', fontweight='bold')
    plt.ylabel('Range', fontweight='bold')
    plt.show()


# show_boxplot(cleaned_data)


# Remove Outliers
def remove_outliers(data):
    df = data.copy()
    for col in list(df.columns):
        # Compute interquantile range
        Q1 = df[str(col)].quantile(0.05)
        Q3 = df[str(col)].quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[str(col)] >= lower_bound) & (df[str(col)] <= upper_bound)]
    return df


without_outliers = remove_outliers(cleaned_data)
# show_boxplot(without_outliers)
print(without_outliers.shape)

# Rescale the data
data_scaler = StandardScaler()
scaled_data = data_scaler.fit_transform(without_outliers)
print(scaled_data.shape)

# Applying the hierarchical clustering algorithm
complete_clustering = linkage(scaled_data, method='complete', metric='euclidean')
average_clustering = linkage(scaled_data, method='average', metric='euclidean')
single_clustering = linkage(scaled_data, method='single', metric='euclidean')

# plt.figure(figsize=(15, 6))
# dendrogram(complete_clustering)
# plt.title('Dendrogram for complete linkage')
# plt.show()

# plt.figure(figsize=(15, 6))
# dendrogram(average_clustering)
# plt.title('Dendrogram for Average linkage')
# plt.show()

# import sys
# sys.setrecursionlimit(1000000)
# plt.figure(figsize=(15, 6))
# dendrogram(single_clustering)
# plt.title('Dendrogram for Single linkage')
# plt.show()

# Observation of the Average Linkage Clustering
cluster_labels = cut_tree(average_clustering, n_clusters=2).reshape(-1, )
without_outliers['Cluster'] = cluster_labels

sns.boxplot(x='Cluster', y='fico', data=without_outliers)
plt.show()
