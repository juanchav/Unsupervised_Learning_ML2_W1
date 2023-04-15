from sklearn.datasets import fetch_openml
import numpy as np

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist['data']
y = mnist['target']

# Save dataset to local file
np.savez('mnist_dataset.npz', X=X, y=y)
