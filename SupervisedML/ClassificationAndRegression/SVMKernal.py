import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import plotly.express as px

x = np.linspace(-5.0, 5.0, 100)
y = np.sqrt(10 ** 2 - x ** 2)
x = np.hstack([x, -x])
y = np.hstack([y, -y])

x1 = np.linspace(-5.0, 5.0, 100)
y1 = np.sqrt(5 ** 2 - x1 ** 2)
y1 = np.hstack([y1, -y1])
x1 = np.hstack([x1, -x1])

plt.scatter(y, x)
plt.scatter(y1, x1)
plt.show()

df1 = pd.DataFrame(np.vstack([y, x]).T, columns=['X1', 'X2'])
df1['Y'] = 0
df2 = pd.DataFrame(np.vstack([y1, x1]).T, columns=['X1', 'X2'])
df2['Y'] = 1
df = pd.concat([df1, df2], ignore_index=True)
print(df.head(5))

# Independent and Dependent features
X = df.iloc[:, :2]
y = df.Y
print(y)

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(y_train)

classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
print('Accuracy (Linear Kernel):', accuracy)

classifier = SVC(kernel="rbf")
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
print('Accuracy (RBF Kernel):', accuracy)

classifier = SVC(kernel="poly")
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
print('Accuracy (Polynomial Kernel):', accuracy)

classifier = SVC(kernel="sigmoid")
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
print('Accuracy (Sigmoid Kernel):', accuracy)

# Polynomial Kernel
# We need to find components for the Polynomial Kernel
# X1,X2,X1_square,X2_square,X1*X2
df['X1_Square'] = df['X1'] ** 2
df['X2_Square'] = df['X2'] ** 2
df['X1*X2'] = (df['X1'] * df['X2'])
print(df.head())

# Independent and Dependent features
X = df[['X1', 'X2', 'X1_Square', 'X2_Square', 'X1*X2']]
y = df['Y']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train)

fig = px.scatter_3d(df, x='X1', y='X2', z='X1*X2', color='Y')
fig.show()

fig = px.scatter_3d(df, x='X1_Square', y='X1_Square', z='X1*X2', color='Y')
fig.show()

classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
print('Accuracy (Linear Kernel):', accuracy)
