import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from tabulate import tabulate
from MLP_PyTorch import MLP
from RNN import RNN
from logistic_regression import LogisticRegression
from MLP_Numpy import MLPNumpy


def main():
    # Pre-process the data
    X_train, X_test, y_train, y_test = data_to_split()

    # Train and test Logistic Regression
    logreg_accuracy = logisticRegression(X_train, X_test, y_train, y_test)

    # Train and test Multi-Layer Perceptron
    mlp_numpy_accuracy = multilayerPerceptron_numpy(X_train, X_test, y_train, y_test)
    mlp_pytorch_accuracy = multilayerPerceptron_pytorch(X_train, X_test, y_train, y_test)

    # train and test RNN
    rnn_accuracy = rnn_eval(X_train, X_test, y_train, y_test)

    display_accuracy(logreg_accuracy, mlp_numpy_accuracy, mlp_pytorch_accuracy, rnn_accuracy)


def display_accuracy(logreg: float = 0, mlpnp: float = 0, mlppy: float = 0, rnn: float = 0):
    models = ["Logistic Regression", "Multi-Layer Perceptron Numpy", "Multi-Layer Perceptron PyTorch", "RNN"]
    accuracy_scores = [logreg, mlpnp, mlppy, rnn]

    # Create a list of dictionaries to store data
    data = [{"Model": model, "Accuracy (%)": score} for model, score in zip(models, accuracy_scores)]

    # Print out accuracy scores using tabulate
    print(tabulate(data, headers="keys", tablefmt="grid"))


def data_to_split():
    data = pd.read_csv('Training Data/Appended75.csv')
    data = data.drop(data.columns[0], axis=1)

    # Vectorize the sentences using bag-of-words model
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['sentence'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, data['sentiment'], test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.toarray())
    X_test = scaler.transform(X_test.toarray())

    sentiment_map = {'positive': 2, 'negative': 0, 'neutral': 1}
    y_train = y_train.map(sentiment_map)
    y_test = y_test.map(sentiment_map)

    return X_train, X_test, y_train, y_test


def split_to_loader(X_train, X_test, y_train, y_test):
    batch_size = 64

    series = pd.Series(X_train.tolist()).apply(np.array)
    train_df = pd.concat([series, y_train.reset_index(drop=True)], axis=1)
    series = pd.Series(X_test.tolist()).apply(np.array)
    test_df = pd.concat([series, y_test.reset_index(drop=True)], axis=1)

    train_df.columns = ['sentence', 'sentiment']
    test_df.columns = ['sentence', 'sentiment']

    train_dataset = MyDataset(train_df)
    test_dataset = MyDataset(test_df)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dl, test_dl


def logisticRegression(X_train, X_test, y_train, y_test):
    # Instantiate the logistic regression model
    logreg = LogisticRegression()

    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_onehot = lb.fit_transform(y_train)

    # Train the model
    logreg.fit(X_train, y_onehot)

    # Predict on the test set
    y_pred = logreg.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred) * 100

    with open('logreg_model.pickle', 'wb') as f:
        pickle.dump(logreg, f)

    return accuracy


def train(model, train_dl, n_epochs, optimizer, criterion, batch_size=64):
    # Switch to GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.train()

    for epoch in range(n_epochs):

        for i, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


def multilayerPerceptron_numpy(X_train, X_test, y_train, y_test):
    mlp = MLPNumpy(num_epochs=45)

    mlp.train(X_train, y_train.values)

    predictions = mlp.predict(X_test)

    accuracy = accuracy_score(y_test, predictions) * 100

    with open('mlp_numpy_model.pickle', 'wb') as f:
        pickle.dump(mlp, f)

    # print(classification_report(y_test, predictions))

    return accuracy


def test(model, test_dl, criterion, batch_size=64):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total

    return accuracy


def multilayerPerceptron_pytorch(X_train, X_test, y_train, y_test):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dl, test_dl = split_to_loader(X_train, X_test, y_train, y_test)

    train_dl_iter = iter(train_dl)
    while True:
        try:
            batch = next(train_dl_iter)
        except StopIteration:
            break

    mlp = MLP(X_train.shape[1])
    mlp = mlp.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    train(mlp, train_dl, 10, optimizer, criterion)

    accuracy = test(mlp, test_dl, criterion)

    with open('mlp_pytorch_model.pickle', 'wb') as f:
        pickle.dump(mlp, f)

    return accuracy


def rnn_eval(X_train, X_test, y_train, y_test):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dl, test_dl = split_to_loader(X_train, X_test, y_train, y_test)

    train_dl_iter = iter(train_dl)
    while True:
        try:
            batch = next(train_dl_iter)
        except StopIteration:
            break

    rnn = RNN(X_train.shape[1])
    rnn = rnn.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
    train(rnn, train_dl, 10, optimizer, criterion)

    accuracy = test(rnn, test_dl, criterion)

    with open('rnn_model.pickle', 'wb') as f:
        pickle.dump(rnn, f)

    return accuracy


if __name__ == "__main__":
    main()
