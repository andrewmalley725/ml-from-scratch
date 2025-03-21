class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        import numpy as np
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.weights = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y)) #fits using closed form solution
        return self.weights
    
    def predict(self, X):
        import numpy as np
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        y = X @ self.weights
        return y