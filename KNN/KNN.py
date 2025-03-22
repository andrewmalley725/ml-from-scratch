class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X =None
        self.y = None

    def compute_G(self, X, Z):
        import numpy as np
        return np.dot(X, Z.T)
    
    def calculate_S(self, X, n, m):
        import numpy as np
        assert n == X.shape[0]
        row_norms = np.sum(X**2, axis=1)
        row_norms = row_norms.reshape(-1,1)
        return np.repeat(row_norms,m,axis=1)
    
    def calculate_R(self, Z, n, m):
        import numpy as np
        assert m == Z.shape[0]
        norms = np.sum(Z**2,axis=1)
        norms = norms.reshape(-1,1)
        return np.repeat(norms,n,axis=1).T
    
    def l2_distance(self, X, Z):
        import numpy as np
        n = X.shape[0]
        m = Z.shape[0]
        G = self.compute_G(X, Z)
        S = self.calculate_S(X, n, m)
        R = self.calculate_R(Z, n, m)
        D = S + R - 2*G
        D = np.maximum(D, 0)
        D = np.sqrt(D)
        return D
    
    def fit(self, X, y):
        import numpy as np
        self.X = X
        self.y = y.flatten()
    
    def predict(self, Z):
        import numpy as np
        from scipy.stats import mode
        D = self.l2_distance(self.X, Z)
        n_test = Z.shape[0]
        preds = np.zeros(n_test)
        for i in range(n_test):
            nearest_indices = np.argsort(D[i, :])[:self.k]
            nearest_labels = self.y[nearest_indices]
            preds[i] = mode(nearest_labels, keepdims=True).mode[0]
        return preds
        
        