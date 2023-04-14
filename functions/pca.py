import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # calculate the covariance matrix
        cov = np.cov(X.T)

        # calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort the eigenvalues in descending order
        sort_index = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_index]

        # select the top n_components eigenvectors
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, :self.n_components]

        self.components = eigenvectors

    def transform(self, X):
        # center the data
        X = X - self.mean

        # project the data onto the eigenvectors
        transformed = np.dot(X, self.components)

        return transformed

    def fit_transform(self, X):
        self.fit(X)
        transformed = self.transform(X)

        return transformed




		




