import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('Training Data/Cleaned_Data.csv')

# Encode the sentiment labels
sentiment_map = {'positive': 2, 'negative': 0, 'neutral': 1}
data['sentiment'] = data['sentiment'].map(sentiment_map)

# Split the data
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

tokenizer = get_tokenizer("basic_english")
vocabulary = set()
vocabulary.add("<START>")
vocabulary.add("<END>")
vocabulary.add("<PAD>")

for df in [train_df, test_df]:
    for idx, sentence in df.sentence.items():
        tokens = tokenizer(sentence)
        df.sentence.iat[idx] = tokens
        for word in tokens:
            vocabulary.add(word)

word2id = {word: id for id, word in enumerate(vocabulary)}


def encode_and_pad_wrapper(max_len: int):
    def encode_and_pad(tokens: list[str]):
        start = [word2id["<START>"]]
        end = [word2id["<END>"]]
        pad = [word2id["<PAD>"]]

        if len(tokens) < max_len - 2:  # 2 tokens for <START> and <END>
            n_pads = max_len - 2 - len(tokens)
            encoded = [word2id[token] for token in tokens]
            return start + encoded + end + pad * n_pads
        else:
            encoded = [word2id[token] for token in tokens]
            truncated = encoded[:max_len - 2]
            return start + truncated + end

    return encode_and_pad


train_df.sentence = train_df.sentence.apply(encode_and_pad_wrapper(100))
train_df = train_df.rename(columns={"sentiment": "label", "sentence": "input_ids"})

test_df.sentence = test_df.sentence.apply(encode_and_pad_wrapper(100))
test_df = test_df.rename(columns={"sentiment": "label", "sentence": "input_ids"})


class RNNDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        df.reset_index(drop=True, inplace=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        input_ids = torch.tensor(row["input_ids"])
        label = torch.tensor(row["label"])
        return input_ids, label


train_rnn_ds = RNNDataset(train_df)
test_rnn_ds = RNNDataset(test_df)

batch_size = 64
train_rnn_dl = DataLoader(train_rnn_ds, batch_size=batch_size, shuffle=True, drop_last=True)
test_rnn_dl = DataLoader(test_rnn_ds, batch_size=batch_size, shuffle=False, drop_last=True)

train_dl_iter = iter(train_rnn_dl)
batch = next(train_dl_iter)


class RNN(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 50,
            hidden_size: int = 128,
            n_layers: int = 2,
            n_classes: int = 3
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):  # (bsz, 100)
        x = self.embedding(x)  # (bsz, 100, embedding_dim)
        outputs, _ = self.rnn(x)  # (bsz, 100, hidden_size)
        outputs = outputs[:, -1, :]  # (bsz, hidden_size): get final hidden states only
        outputs = self.fc(outputs)  # (bsz, n_classes)
        return outputs


rnn = RNN(len(word2id))
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
train(rnn, train_rnn_dl, 10, optimizer, criterion)


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

        print(f"Test Loss: {test_loss:.5f} | Accuracy: {accuracy:.2f}%")

    return accuracy


# After training, call the test function to evaluate the trained model on the test data
test_accuracy = test(rnn, test_rnn_dl, criterion)
