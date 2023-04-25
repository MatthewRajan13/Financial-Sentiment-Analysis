import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from RNN import RNN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from logistic_regression import LogisticRegression
from mlp import MLP
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from MyDataset import MyDataset


def main():
    # Pre-process the data
    X_train, X_test, y_train, y_test = preprocess()

    # Train and test Logistic Regression
    logisticRegression(X_train, X_test, y_train, y_test)

    # Train and test Multi-Layer Perceptron
    # multilayerPerceptron(X_train, X_test, y_train, y_test)

    # train and test RNN
    rnn_eval(X_train, X_test, y_train, y_test)


def preprocess():
    data = pd.read_csv('Training Data/Appended75.csv')
    data = data.drop(data.columns[0], axis=1)

    # Get Set of words
    tokenizer = get_tokenizer("basic_english")
    vocabulary = set()
    for idx, sentence in data.sentence.items():
        tokens = tokenizer(sentence)
        for word in tokens:
            vocabulary.add(word)

    # Vectorize the sentences using bag-of-words model
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['sentence'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, data['sentiment'], test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.toarray())
    X_test = scaler.transform(X_test.toarray())

    return X_train, X_test, y_train, y_test


def logisticRegression(X_train, X_test, y_train, y_test):
    # Instantiate the logistic regression model
    logreg = LogisticRegression()

    # Encode the sentiment labels
    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
    y_train = y_train.map(sentiment_map)
    y_test = y_test.map(sentiment_map)

    # Train the model
    logreg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = logreg.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print('Logistic Regression Accuracy:', accuracy)


def multilayerPerceptron(X_train, X_test, y_train, y_test):
    # Get sizes and classes
    num_dim, num_features = X_train.shape
    _, num_test = X_test.shape
    classes = np.unique(y_train)
    num_classes = len(classes)

    # Set model and Learning Rate
    model = {'weights': np.zeros((num_dim, num_classes))}
    learning_rate = 1

    # TODO: COMPLETE MLP
    # Instantiate the MLP model
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 3
    mlp_model = MLP(input_size, hidden_size, output_size)

    num_epochs = 1000
    learning_rate = 0.001

    mlp_model.train(X_train, y_train, num_epochs, learning_rate)

    # Forward pass on testing data
    test_pred_probs = mlp_model.forward(X_test)

    # Convert predicted probabilities to predicted labels
    test_preds = np.argmax(test_pred_probs, axis=1)

    # Calculate accuracy
    accuracy = np.mean(test_preds == y_test)

    print("Accuracy: {:.2f}%".format(accuracy * 100))


def rnn_eval(X_train, X_test, y_train, y_test):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    batch_size = 64

    series = pd.Series(X_train.tolist()).apply(np.array)
    train_df = pd.concat([series, y_train.reset_index(drop=True)], axis=1)
    series = pd.Series(X_test.tolist()).apply(np.array)
    test_df = pd.concat([series, y_test.reset_index(drop=True)], axis=1)

    train_df.columns = ['sentence', 'sentiment']
    test_df.columns = ['sentence', 'sentiment']

    sentiment_map = {'positive': 2, 'negative': 0, 'neutral': 1}
    train_df.sentiment = train_df.sentiment.map(sentiment_map)
    test_df.sentiment = test_df.sentiment.map(sentiment_map)

    train_dataset = MyDataset(train_df)
    test_dataset = MyDataset(test_df)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    train_dl_iter = iter(train_dl)
    while True:
        try:
            batch = next(train_dl_iter)
        except StopIteration:
            break

    rnn = RNN(X_train.shape[1])
    rnn = rnn.to(device)

    def train(model, train_dl, n_epochs, optimizer, criterion, batch_size=64):
        model.train()

        for epoch in range(n_epochs):
            train_loss = 0.0

            for i, (inputs, labels) in enumerate(train_dl):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            train_loss = train_loss / (len(train_dl.dataset) // batch_size)

            print(f"Epoch {epoch + 1}: Loss {train_loss:.5f}")

        print("Finished training.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
    train(rnn, train_dl, 10, optimizer, criterion)

    def test(model, test_dl, criterion, batch_size=64):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_dl:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            test_loss = test_loss / (len(test_dl.dataset) // batch_size)
            accuracy = 100.0 * correct / total

            print(f"Test Loss: {test_loss:.5f} | RNN Accuracy: {accuracy:.2f}%")

        return accuracy

    # After training, call the test function to evaluate the trained model on the test data
    test_accuracy = test(rnn, test_dl, criterion)


if __name__ == "__main__":
    main()
