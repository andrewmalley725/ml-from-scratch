class NaiveBayes:
    def __init__(self):
        self.y = None
        self.X = None

    def PY(self):
        import numpy as np
        y = np.concatenate([self.y, [-1,1]]) #smoothing
        pos = np.sum(y == 1) / len(y)
        neg = np.sum(y == -1) / len(y)
        return pos, neg
    
    def PXY(self):
        import numpy as np
        d = self.X.shape[1]
        #add points to X and y for smoothing
        X = np.concatenate([self.X, np.ones((2,d)), np.zeros((2,d))])
        y = np.concatenate([self.y, [-1,1,-1,1]])
        posbrob = np.sum(X[y == 1], axis=0) / np.sum(y == 1)
        negbrob = np.sum(X[y == -1], axis=0) / np.sum(y == -1)
        return posbrob, negbrob
    
    def log_likelihood(self, X):
        import numpy as np
        pos, neg = self.PY()
        posprob, negprob = self.PXY()
        log_pos = np.log(pos)
        log_neg = np.log(neg)
        log_likelihood_pos = X @ np.log(posprob) + (1 - X) @ np.log(1 - posprob)
        log_likelihood_neg = X @ np.log(negprob) + (1 - X) @ np.log(1 - negprob)
        log_ratio = (log_likelihood_pos + log_pos) - (log_likelihood_neg + log_neg)
        predictions = np.where(log_ratio > 0, 1, -1)
        return predictions
    
    def fit(self, X, y):
        self.y = y
        self.X = X
        predictions = self.log_likelihood(X)
        return predictions
    
    def predict(self, X):
        predictions = self.log_likelihood(X)
        return predictions
        