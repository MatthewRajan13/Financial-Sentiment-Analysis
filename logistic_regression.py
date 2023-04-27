import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=400):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.theta = np.zeros((X.shape[1], 3))

        for i in range(self.n_iter):
            z = np.dot(X, self.theta)
            h = self.softmax(z)
            gradient = np.dot(X.T, (h - y)) / len(X)
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        z = np.dot(X, self.theta)
        h = self.softmax(z)
        predictions = np.argmax(h, axis=1)
        return predictions

    def compute_loss(self, X, y):
        z = np.dot(X, self.theta)
        h = self.softmax(z)
        loss = -np.mean(np.sum(y * np.log(h), axis=1))
        return loss

    def softmax(self, z):
        exp_z = np.exp(z)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)