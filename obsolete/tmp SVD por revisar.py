class SVD:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None

    def fit(self, X):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        if self.n_components is not None:
            self.components = Vt[:self.n_components]
        else:
            self.components = Vt
        return U, s, Vt

    def transform(self, X):
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
       
       
svd=SVD(n_components=10)
from PIL import Image
U, s, Vt =svd.fit(myPhoto)
#U, s, Vt = np.linalg.svd(myPhoto)
#U, s, Vt =svd.fit_transform(myPhoto)

#reconstructed_img = Image.fromarray(myPhoto_transformed.astype(np.uint8))
n_components = 3
reconstructed_gray = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
reconstructed_img = cv2.cvtColor(reconstructed_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
