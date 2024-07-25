import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

"""Cancer Dataset"""
print('Cancer Dataset')
cancer_dataset = load_breast_cancer()
print(cancer_dataset.keys())
# print(cancer_dataset['DESCR'])
# print(cancer_dataset.DESCR)

df = pd.DataFrame(cancer_dataset['data'], columns=cancer_dataset['feature_names'])
print(df.head())

# PCA Visualization
# It is difficult to visualize high dimensional data, we can use PCA to find the first two principal components, and
# visualize the data in this new, two-dimensional space, with a single scatter-plot. Before we do this though,
# we'll need to scale our data so that each feature has a single unit variance.

# Standardization
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
print('Data after scaling:\n', scaled_data)
print('Shape of scaled data:', scaled_data.shape)

# Applying PCA Algorithms
pca = PCA(n_components=2)
data_pca = pca.fit_transform(scaled_data)  # Transformed the all 30 columns into 2 columns
print('Dimensionality reduction after applying PCA algorithm:\n', data_pca)
print('Shape of data after applying PCA:', data_pca.shape)

print('Explained variance (eigenvalues) of the principal components:', pca.explained_variance_)
print('Explained variance ratio of the principal components:', pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cancer_dataset['target'], cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.title('Principal Component Analysis (n_components=2)')
plt.show()

# Interpreting the components
print(pca.components_)

df_comp = pd.DataFrame(pca.components_, columns=cancer_dataset['feature_names'])

plt.figure(figsize=(12, 6))
sns.heatmap(df_comp, cmap='plasma', )
plt.show()

"""IRIS Dataset"""
print('\nIRIS Dataset')
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Create a PCA object and fit the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)

# Print the explained variance ratio of the selected components
print('Explained variance (eigenvalues) of the principal components:', pca.explained_variance_)
print('Explained variance ratio of the principal components:', pca.explained_variance_ratio_)

# Choosing The Number of Components
# You can choose the number of components based on the explained variance ratio. To retain 95% of the variance:
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print('Number of Components:', n_components)

# Plot the transformed data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Applying LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy_pca = model.score(X_test, y_test)

X_train_original, X_test_original, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_original = LogisticRegression()
model_original.fit(X_train_original, y_train)
accuracy_original = model_original.score(X_test_original, y_test)

print("Accuracy with PCA:", accuracy_pca)
print("Accuracy without PCA:", accuracy_original)

"""Wine Dataset"""
print('\nWine Dataset')
wine_data = load_wine()
print(wine_data.keys())
print(wine_data['feature_names'])
print(wine_data['target_names'])

df = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
print(df.head())
print(df.iloc[:, 1:].describe())

# Standardization
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
print('Data after scaling:\n', scaled_data)
print('Shape of scaled data:', scaled_data.shape)

# Applying PCA Algorithms
pca = PCA(n_components=2)
data_pca = pca.fit_transform(scaled_data)
print('Dimensionality reduction after applying PCA algorithm:\n', data_pca[:20, :])
print('Shape of data after applying PCA:', data_pca.shape)

print('Explained variance (eigenvalues) of the principal components:', pca.explained_variance_)
print('Explained variance ratio of the principal components:', pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=wine_data['target'], cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.title('Principal Component Analysis (n_components=2)')
plt.show()

# Interpreting the components
print(pca.components_)

df_comp = pd.DataFrame(pca.components_, columns=wine_data['feature_names'])

plt.figure(figsize=(12, 6))
sns.heatmap(df_comp, cmap='plasma', )
plt.show()
