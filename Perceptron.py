class Perceptron:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        import numpy as np
        n,d = X.shape
        self.weights = np.zeros(d) #init weights as a d dimensional 0 vector
        self.bias = 0.0
        w = np.append(self.weights, self.bias) #append bias to weights
        X = np.hstack((X, np.ones((n, 1)))) #append 1 to each row of X
        iter = 0
        while True:
            iter += 1
            m = 0 #number of misclassified points
            for i in range(n):
                if y[i] * np.dot(w,X[i]) <= 0: #if signs of true label and predicted label are different
                    w += y[i] * X[i] #move hyperplane to correctly classify point
                    m += 1
            if m == 0: #if no misclassified points, break
                break
        self.weights = w[:-1]
        self.bias = w[-1]
        return self.weights, self.bias, iter
    
    def predict(self, X):
        import numpy as np
        n,d = X.shape
        w = np.append(self.weights, self.bias)
        X = np.hstack((X, np.ones((n, 1))))
        y_pred = np.sign(np.dot(X, w)) #sign determine side of hyperplane
        return y_pred




        

