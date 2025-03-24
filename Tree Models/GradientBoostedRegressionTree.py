import numpy as np
from RegressionTree import RegressionTree

class GBRT:
    def __init__(self, num_trees=20, alpha=0.1, max_depth=4):
        self.trees = []
        self.num_trees = num_trees
        self.alpha = alpha
        self.max_depth = max_depth
        self.initial_pred = None

    def fit(self, X, y):
        n = X.shape[0]
        self.initial_pred = starting_leaf = np.mean(y)
        new_preds = np.full(n, starting_leaf)
        resids = y - new_preds
        for _ in range(self.num_trees):
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X, resids)
            self.trees.append(tree)
            preds = tree.predict(X)
            new_preds += self.alpha * preds
            resids = y - new_preds

    def predict(self, X):
        preds = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            preds += self.alpha * tree.predict(X)
        return preds