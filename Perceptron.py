class Perceptron:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        import numpy as np
        n,d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0
        w = np.append(self.weights, self.bias)
        X = np.hstack((X, np.ones((n, 1))))
        iter = 0
        while True:
            iter += 1
            m = 0
            for i in range(n):
                if y[i] * np.dot(w,X[i]) <= 0:
                    w += y[i] * X[i]
                    m += 1
            if m == 0:
                break
        self.weights = w[:-1]
        self.bias = w[-1]
        return self.weights, self.bias
    
    def predict(self, X):
        import numpy as np
        n,d = X.shape
        w = np.append(self.weights, self.bias)
        X = np.hstack((X, np.ones((n, 1))))
        y_pred = np.sign(np.dot(X, w))
        return y_pred




        

