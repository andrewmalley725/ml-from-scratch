import numpy as np
from RegressionTree import RegressionTree

class RandomForestRegressor:
    def __init__(self, num_trees, max_depth=np.inf):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X, y):
        n = len(X)
        for _ in range(self.num_trees):
            idx = np.random.choice(n, size=n, replace=True)
            X_sub = X[idx]
            y_sub = y[idx]
            clf = RegressionTree(max_depth=self.max_depth)
            clf.fit(X_sub, y_sub)
            self.trees.append(clf)
        return self.trees
    
    def predict(self, X):
        preds = np.zeros(len(X))
        for tree in self.trees:
            pred = tree.predict(X)
            preds += pred
        preds /= len(self.trees)
        return preds