import numpy as np
from RegressionTree import RegressionTree

class GBC:
    def __init__(self, num_trees=50, alpha=0.1, max_depth=4):
        self.trees = []
        self.num_trees = num_trees
        self.alpha = alpha
        self.max_depth = max_depth
        self.initial_pred = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n = X.shape[0]
        p = np.mean(y)
        self.initial_pred = np.log(p/(1-p))
        preds = np.full(n, self.initial_pred) #log odds
        probs = self._sigmoid(preds)
        resids = y - probs
        for _ in range(self.num_trees):
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X,resids)
            self.trees.append(tree)
            temp_preds = tree.predict(X)
            preds += temp_preds * self.alpha
            probs = self._sigmoid(preds)
            resids = y - probs

    def predict(self, X):
        preds = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            preds += self.alpha * tree.predict(X)
        probs = self._sigmoid(preds)
        res = np.where(probs >= .5,1,0)
        return res
