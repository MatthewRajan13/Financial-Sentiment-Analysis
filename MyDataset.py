import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sentence = self.data.iloc[index]['sentence']
        sentiment = self.data.iloc[index]['sentiment']
        return sentence, sentiment

    def __len__(self):
        return len(self.data)