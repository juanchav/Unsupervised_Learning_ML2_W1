import numpy as np

class TSNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=200):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.embedding_ = None

    def fit(self, X):
        # Compute pairwise similarities
        similarities = self._compute_pairwise_similarities(X)

        # Initialize embedding with small random values
        np.random.seed(0)
        embedding = np.random.randn(X.shape[0], self.n_components) * 1e-4

        # Perform gradient descent to optimize embedding
        for i in range(1000):
            embedding_grad = self._compute_embedding_grad(embedding, similarities)
            embedding = self._update_embedding(embedding, embedding_grad)

        self.embedding_ = embedding

    def transform(self, X):
        # Compute pairwise similarities between X and training data
        similarities = self._compute_pairwise_similarities(X, self.embedding_)

        # Compute embedding of X using training data embedding and learned similarities
        embedding = np.random.randn(X.shape[0], self.n_components) * 1e-4
        for i in range(1000):
            embedding_grad = self._compute_embedding_grad(embedding, similarities)
            embedding = self._update_embedding(embedding, embedding_grad)

        return embedding

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_

    def _compute_pairwise_similarities(self, X, Y=None):
        if Y is None:
            Y = X
        # Compute Euclidean distances between all pairs of points
        #distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2*np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T
        distances_sq = np.sum(X**2, axis=1) - 2*np.dot(X, Y.T) + np.sum(Y**2, axis=1).T
        # Compute Gaussian similarities
        similarities = np.exp(-distances_sq / (2*self.perplexity**2))
        return similarities

    def _compute_embedding_grad(self, embedding, similarities):
        # Compute pairwise differences in the embedding
        differences = embedding[:, np.newaxis, :] - embedding[np.newaxis, :, :]
        # Compute distances between all pairs of points in the embedding
        distances_sq = np.sum(differences**2, axis=-1)
        # Compute Q matrix (Student-t distribution)
        Q = 1 / (1 + distances_sq)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        # Compute P matrix (Gaussian distribution)
        P = similarities / np.sum(similarities)
        # Compute gradient of KL divergence
        grad = 4 * np.dot((P - Q) * Q, differences)
        return grad

    def _update_embedding(self, embedding, embedding_grad):
        # Perform gradient descent to update embedding
        embedding -= self.learning_rate * embedding_grad
        return embedding
