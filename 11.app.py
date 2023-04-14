from sklearn.datasets import fetch_openml
from functions.svd import SVD
import matplotlib.pyplot as plt


mnist = fetch_openml('mnist_784',)
X, y = mnist["data"], mnist["target"]


plt.imshow(X[:1])
plt.show()