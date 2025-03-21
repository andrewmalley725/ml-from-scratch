class LogisticRegression:
    def __init__(self, alpha=0.01, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        import numpy as np
        return 1 / (1 + np.exp(-z))
    
    def y_pred(self, X):
        import numpy as np
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def log_loss(self, X, y):
        import numpy as np
        y = (y + 1) / 2
        y_pred = self.y_pred(X)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def gradients(self, X, y):
        import numpy as np
        n, d = X.shape
        z = -y * (np.dot(X, self.weights) + self.bias)
        sigma = self.sigmoid(z)
        wgrad = (-y * sigma) @ X
        bgrad = np.sum(-y * sigma)
        return wgrad, bgrad
    
    def fit_gd(self, X, y):
        import numpy as np
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0
        losses = np.zeros(self.max_iter)
        wgrad, bgrad = self.gradients(X, y)
        for i in range(self.max_iter):
            self.weights -= self.alpha * wgrad
            self.bias -= self.alpha * bgrad
            losses[i] = self.log_loss(X, y)
            wgrad, bgrad = self.gradients(X, y)
        return self.weights, self.bias, losses