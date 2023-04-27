import numpy as np


class MLPNumpy:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.num_classes = 3

    def train(self, X, y):
        self.X = X
        self.y = y

        # Add bias
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        # Initialize weight matrix with zeros
        w = np.zeros((self.num_classes, X.shape[1]))

        for epoch in range(self.num_epochs):
            for i in range(X.shape[0]):
                scores = np.dot(w, X[i])

                y_pred = np.argmax(scores)
                y_true = y[i]

                if y_pred != y_true:
                    w[y_true] += X[i]
                    w[y_pred] -= X[i]

        self.w = w

    def predict(self, X):
        self.X = X
        # Add bias
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        scores = np.dot(self.w, X.T)

        y_pred = np.argmax(scores, axis=0)

        return y_pred
