import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Define logistic regression model
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.theta = np.zeros(X.shape[1])

        for i in range(self.n_iter):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / len(X)
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        z = np.dot(X, self.theta)
        h = sigmoid(z)
        predictions = [1 if p >= 0.5 else 0 for p in h]
        return predictions

    def compute_loss(self, X, y):
        z = np.dot(X, self.theta)
        h = sigmoid(z)
        loss = -(y * np.log(h) + (1 - y) * np.log(1 - h)).mean()
        return loss
