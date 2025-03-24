import numpy as np

class RegressionTree:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature      # Feature index to split on
            self.threshold = threshold  # Split value
            self.left = left           # Left child
            self.right = right         # Right child
            self.value = value         # Mean prediction if leaf

    def _sqloss(self, y):
        n = len(y)
        y_hat = np.sum(y) / n
        return np.sum((y - y_hat) ** 2)
    
    def _select_cut(self, X, y):
        m = len(X)
        sorted_X = np.sort(X)
        best_cut = None
        best_loss = np.inf
        for i in range(1,m):
            cut = (sorted_X[i] + sorted_X[i-1]) / 2
            left_wing = np.where(X < cut)[0]
            right_wing = np.where(X >= cut)[0]
            y_left = y[left_wing]
            y_right = y[right_wing]
            if len(y_left) > 0 and len(y_right) > 0:
                left_loss = self._sqloss(y_left)
                right_loss = self._sqloss(y_right)
                total_loss = left_loss + right_loss
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_cut = cut
        return best_cut, best_loss
    
    def _select_first_node(self, X, y):
        n, d = X.shape
        best_impurity = np.inf
        best_cut = None
        best_feature = None
        for i in range(d):
            feature = X[:, i]
            temp_cut, loss = self._select_cut(feature, y)
            if loss < best_impurity:
                best_impurity = loss
                best_cut = temp_cut
                best_feature = i
        return best_feature, best_cut, best_impurity

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        n_samples = len(y)
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           np.var(y) < 1e-6:  # Very small variance
            return self.Node(value=np.mean(y))
        best_feature, best_cut, best_loss = self._select_first_node(X, y)
        if best_feature is None or np.isinf(best_loss):
            return self.Node(value=np.mean(y))
        left_idx = X[:, best_feature] < best_cut
        right_idx = X[:, best_feature] >= best_cut
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return self.Node(value=np.mean(y))
        left_child = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return self.Node(best_feature, best_cut, left_child, right_child)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)