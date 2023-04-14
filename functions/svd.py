import numpy as np

class SVD:

    def __init__(self,n_vectors):   
        self.n_vectors=n_vectors 

    def fit(self,x):
        '''Creates the matrixes for SVD transformation and generates the truncate matrix, which allows
        to reduce new features using
        params used:
        x: Data to train
        n_vectors: How many vectors you will use
        ''' 
        self.x=x  
        #compute the vectors
        self.U, self.s, self.Vt = np.linalg.svd(self.x) 
        #take the n_components we need
        self.Uk = self.U[:, :self.n_vectors]
        self.sk = np.diag(self.s[:self.n_vectors])
        self.Vk = self.Vt[:self.n_vectors, :]    
        #compute mean and std to standarization    
        self.mu = np.mean(self.x, axis=0)
        self.sigma = np.std(self.x, axis=0)
        #compute truncate svd
        # self.truncate_svd = (Uk@ np.diag(sk)@ Vk)
        self.truncate_svd=self.Vk.T

    def transform(self,x):   
        # X_new_centered = x - self.mu
        # X_new_scaled = X_new_centered / self.sigma
        # return np.dot(X_new_scaled, self.truncate_svd)
        return ( x@ self.truncate_svd)
    
    def fit_transform(self,x):
        self.x=x  
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self): 
        X_reconstructed = ( self.x@ self.truncate_svd).dot(self.truncate_svd.T) + np.mean(self.x, axis=0)      
        return X_reconstructed


class SVD1:
    def __init__(self, n_components=None):
        self.n_components = n_components
        
    def fit(self, X):
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)
        self.U = U[:, :self.n_components]
        self.sigma = sigma[:self.n_components]
        self.VT = VT[:self.n_components, :]
        
    def fit_transform(self, X):
        self.fit(X)
        X_transformed = self.U @ np.diag(self.sigma)
        return X_transformed
    
    def transform(self, X):
        X_transformed = self.U @ np.diag(self.sigma)
        return X_transformed


class SVD2:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None

    def fit(self, X):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        if self.n_components is not None:
            self.components = Vt[:self.n_components]
        else:
            self.components = Vt

    def transform(self, X):
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
