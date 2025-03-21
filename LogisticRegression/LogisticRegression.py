class LogisticRegression:
    def __init__(self, alpha=0.01, max_iter=1000, SGD=False):
        self.alpha = alpha
        self.max_iter = max_iter
        self.SGD = SGD
        self.weights = None
        self.bias = None

    def sigmoid(self, z): #sigmoid function interpreted as probability between 0 and 1
        import numpy as np
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        import numpy as np
        z = np.dot(X, self.weights) + self.bias #z = wx + b then passed into sigmoid to get probability
        return self.sigmoid(z)
    
    def log_loss(self, X, y):
        import numpy as np
        y = (y + 1) / 2
        y_pred = self.predict(X)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def gradients(self, X, y): #calculates gradients of log loss with respect to weights and bias
        import numpy as np
        n, d = X.shape
        z = -y * (np.dot(X, self.weights) + self.bias)
        sigma = self.sigmoid(z)
        wgrad = np.dot(-y * sigma, X)
        bgrad = np.sum(-y * sigma)
        return wgrad, bgrad
    
    def fit(self, X, y):
        import numpy as np
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0
        losses = np.zeros(self.max_iter)
        if self.SGD == False: #Batch Gradient Descent
            for i in range(self.max_iter):
                wgrad, bgrad = self.gradients(X, y) #takes gradients of whole dataset
                self.weights -= self.alpha * wgrad
                self.bias -= self.alpha * bgrad
                losses[i] = self.log_loss(X, y)
        else: #Stochastic Gradient Descent
            for i in range(self.max_iter):
                idx = np.random.randint(0, n)
                wgrad, bgrad = self.gradients(X[idx].reshape(1,-1), np.array([y[idx]])) #takes gradients of one random point
                self.weights -= self.alpha * wgrad
                self.bias -= self.alpha * bgrad
                losses[i] = self.log_loss(X, y)
        return self.weights, self.bias, losses