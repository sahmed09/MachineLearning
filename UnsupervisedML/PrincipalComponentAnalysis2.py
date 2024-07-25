import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

breast = load_breast_cancer()
breast_data = breast.data
breast_labels = breast.target
print(breast_data.shape, breast_labels.shape)

labels = np.reshape(breast_labels, (569, 1))
print(labels.shape)

final_breast_data = np.concatenate([breast_data, labels], axis=1)
print(final_breast_data.shape)

breast_dataset = pd.DataFrame(final_breast_data)
print(breast_dataset.head())

features = breast.feature_names
features_labels = np.append(features, 'label')
print(features_labels)

breast_dataset.columns = features_labels
print(breast_dataset.head())

breast_dataset['label'].replace(0, 'Benign', inplace=True)
breast_dataset['label'].replace(1, 'Malignant', inplace=True)
print(breast_dataset.tail())

x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)  # normalizing the features
print(x.shape)

feat_cols = ['feature' + str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x, columns=feat_cols)
print(normalised_breast.tail())

pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)

principal_breast_Df = pd.DataFrame(data=principalComponents_breast,
                                   columns=['principal component 1', 'principal component 2'])
print(principal_breast_Df.tail())

print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

plt.figure(figsize=(8, 8))
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Principal Component - 1', fontsize=15)
plt.ylabel('Principal Component - 2', fontsize=15)
plt.title("Principal Component Analysis of Breast Cancer Dataset", fontsize=15)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets, colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1'],
                principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
plt.legend(targets, prop={'size': 12})
plt.show()

cancer = load_breast_cancer(as_frame=True)
df = cancer.frame  # creating dataframe
print(df.head())
print('Original Dataframe shape :', df.shape)

X = df[cancer['feature_names']]
print('Inputs Dataframe shape   :', X.shape)

scaled = StandardScaler().fit_transform(X)
scaled_df = pd.DataFrame(scaled, columns=X.columns)
print(scaled_df)

covariance = scaled_df.cov()
plt.figure(figsize=(10, 8))
sns.heatmap(covariance)
plt.show()

eigenvalues, eigenvectors = np.linalg.eig(covariance)
print('Eigen values:\n', eigenvalues)
print('Eigen values Shape:', eigenvalues.shape)
print('Eigen Vector Shape:', eigenvectors.shape)

idx = eigenvalues.argsort()[::-1]  # Index the eigenvalues in descending order
eigenvalues = eigenvalues[idx]  # Sort the eigenvalues in descending order
eigenvectors = eigenvectors[:, idx]  # sort the corresponding eigenvectors accordingly

explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
print(explained_var)

n_components = np.argmax(explained_var >= 0.50) + 1
print(n_components)

# PCA component or unit matrix
u = eigenvectors[:, :n_components]
pca_component = pd.DataFrame(u, index=cancer['feature_names'], columns=['PC1', 'PC2'])

# plotting heatmap
plt.figure(figsize=(5, 7))
sns.heatmap(pca_component)
plt.title('PCA Component')
plt.show()

pca = PCA(n_components=2)
x_pca = pca.fit_transform(scaled_df)

# Create the dataframe
df_pca1 = pd.DataFrame(x_pca, columns=['PC{}'.format(i + 1) for i in range(n_components)])
print(df_pca1)

plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cancer['target'], cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

# components
print(pca.components_)
