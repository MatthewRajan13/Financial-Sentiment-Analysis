import torch
from RNN import RNN
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import pickle


def main():
    # TODO: Predict labels
    data = preprocess()


def preprocess():
    data = pd.read_csv('Training Data/S&P_News.csv')
    data = data['Title']

    # Vectorize the sentences using bag-of-words model
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(data)

    scaler = StandardScaler()
    data = scaler.fit_transform(data.toarray())

    return torch.tensor(data)


if __name__ == "__main__":
    main()
