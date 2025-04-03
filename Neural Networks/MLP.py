import numpy as np

class MLP:
    def __init__(self, specs, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.specs = specs
        self.W, self.b = self.init_weights()

    def relu(self, x):
        return np.maximum(0, x)

    def relu_gradient(self, x):
        return np.where(x > 0, 1, 0)

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    def init_weights(self):
        W = []
        b = []
        for i in range(len(self.specs) - 1):
            W.append(np.random.randn(self.specs[i], self.specs[i + 1]) * np.sqrt(2.0 / self.specs[i]))
            b.append(np.zeros((1, self.specs[i + 1])))
        return W, b

    def forward(self, X):
        X = np.asarray(X)  # Convert to NumPy array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        W, b = self.W, self.b
        A = [X]
        Z = [X]
        L = len(W)
        for l in range(L):
            z = A[l] @ W[l] + b[l]  # Matrix multiplication followed by bias addition
            Z.append(z)
            if l < L - 1:
                a = self.relu(z)
            else:
                a = z  # No activation on the output layer
            A.append(a)
        return A, Z

    def backward(self, A, Z, y):
        y = np.asarray(y)  # Convert y to NumPy array
        if y.ndim == 1:
            y = y.reshape(-1, 1)  # Ensure y is 2D
        W, b = self.W, self.b
        L = len(W)
        dW = []
        db = []
        delta = self.mse_gradient(y, A[-1])
        for l in range(L - 1, -1, -1):
            dW.insert(0, A[l].T @ delta)
            db.insert(0, np.sum(delta, axis=0, keepdims=True))
            if l > 0:
                delta = (delta @ W[l].T) * self.relu_gradient(Z[l])
        return dW, db

    def fit(self, X, y):
        X = np.asarray(X)  # Ensure X is a NumPy array
        y = np.asarray(y)  # Ensure y is a NumPy array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        W, b = self.W, self.b
        for epoch in range(self.num_iterations):
            A, Z = self.forward(X)
            dW, db = self.backward(A, Z, y)
            for l in range(len(W)):
                W[l] -= self.learning_rate * dW[l]
                b[l] -= self.learning_rate * db[l]
            if epoch % 100 == 0:
                loss = self.mse(y, A[-1])
                print(f'Epoch {epoch}, Loss: {loss}')
        self.W, self.b = W, b
        return self

    def predict(self, X):
        X = np.asarray(X)  # Ensure X is a NumPy array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        A, _ = self.forward(X)
        return A[-1]