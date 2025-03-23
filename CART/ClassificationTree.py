import numpy as np

class ClassificationTree:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature      # Index of feature to split on
            self.threshold = threshold  # Threshold value for numeric features
            self.left = left           # Left child node
            self.right = right         # Right child node
            self.value = value         # Predicted class if leaf node

    def _gini_numeric(self, X, y):
        m = len(y)
        X_sorted = np.sort(X)
        best_impurity = 1
        best_cut = None
        for i in range(1, m):
            cut = (X_sorted[i] + X_sorted[i - 1]) / 2
            true_indexes = np.where(X < cut)[0]
            false_indexes = np.where(X >= cut)[0]
            true_Y = np.sum(y[true_indexes] == 1)
            true_N = np.sum(y[true_indexes] == 0)
            false_Y = np.sum(y[false_indexes] == 1)
            false_N = np.sum(y[false_indexes] == 0)
            true_total = true_Y + true_N
            false_total = false_Y + false_N
            true_impurity = 1 - (true_Y / true_total)**2 - (true_N / true_total)**2 if true_total > 0 else 0
            false_impurity = 1 - (false_Y / false_total)**2 - (false_N / false_total)**2 if false_total > 0 else 0
            total_impurity = (true_total / m) * true_impurity + (false_total / m) * false_impurity
            if total_impurity < best_impurity:
                best_impurity = total_impurity
                best_cut = cut
        return best_cut, best_impurity
    
    def _gini_impurity(self, X, y):
        m = len(y)
        if len(np.unique(X)) == 2:
            true_Y = np.sum((X == 1) & (y == 1))
            true_N = np.sum((X == 1) & (y == 0))
            false_Y = np.sum((X == 0) & (y == 1))
            false_N = np.sum((X == 0) & (y == 0))
            true_total = true_Y + true_N
            false_total = false_Y + false_N
            true_impurity = 1 - (true_Y / true_total)**2 - (true_N / true_total)**2 if true_total > 0 else 0
            false_impurity = 1 - (false_Y / false_total)**2 - (false_N / false_total)**2 if false_total > 0 else 0
            impurity = (true_total / m) * true_impurity + (false_total / m) * false_impurity
        else:
            impurity = self._gini_numeric(X, y)
        return impurity
    
    def _select_first_node(self, X, y):
        n, m = X.shape
        best_impurity = 1
        best_feature = None
        best_cut = None
        for i in range(m):
            feature = X[:, i]
            impurity_result = self._gini_impurity(feature, y)
            if isinstance(impurity_result, tuple):
                temp, impurity = impurity_result
            else:
                impurity = impurity_result
                temp = None
            if impurity < best_impurity:
                best_impurity = impurity
                best_feature = i
                best_cut = temp
        return best_feature, best_impurity, best_cut

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = self._most_common_class(y)
            return self.Node(value=leaf_value)
        best_feature, best_impurity, best_cut = self._select_first_node(X, y)
        if best_feature is None or best_impurity == 1:
            leaf_value = self._most_common_class(y)
            return self.Node(value=leaf_value)
        feature_values = X[:, best_feature]
        if best_cut is None:  # Binary feature
            left_idx = feature_values == 0
            right_idx = feature_values == 1
        else:  # Numeric feature
            left_idx = feature_values < best_cut
            right_idx = feature_values >= best_cut
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            leaf_value = self._most_common_class(y)
            return self.Node(value=leaf_value)
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return self.Node(best_feature, best_cut, left, right)

    def _most_common_class(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if node.threshold is None:  # Binary feature
            if x[node.feature] == 0:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        else:  # Numeric feature
            if x[node.feature] < node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)