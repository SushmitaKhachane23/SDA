# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# STEP 1: Load Iris Dataset
iris = load_iris()

X = iris.data        # Features
y = iris.target      # Target labels

print("Dataset Shape:", X.shape)


# STEP 2: Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# STEP 3: Apply PCA (Reduce to 2 Components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# STEP 4: Check Explained Variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Preserved:", sum(pca.explained_variance_ratio_))


# STEP 5: Visualize PCA Components
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y,
    cmap='viridis'
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Dimensionality Reduction (Iris Dataset)")
plt.colorbar(label="Species")

plt.grid(True)
plt.show()