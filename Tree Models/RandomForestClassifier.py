import numpy as np
from collections import Counter
from ClassificationTree import ClassificationTree

class RandomForestClassifier:
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
            clf = ClassificationTree(max_depth=self.max_depth)
            clf.fit(X_sub, y_sub)
            self.trees.append(clf)
        return self.trees
    
    def predict(self, X):
        preds = []
        for tree in self.trees:
            pred = tree.predict(X)
            preds.append(pred)
        preds = np.array(preds)
        final_preds = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            votes = preds[:, i]
            final_preds[i] = Counter(votes).most_common(1)[0][0]
        return final_preds

    

