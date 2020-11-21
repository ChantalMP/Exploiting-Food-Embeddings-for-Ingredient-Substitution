import json

import torch
from torch.utils.data import Dataset


class REDataset(Dataset):
    def __init__(self, subset='train'):
        file_name = f'relation_extraction/data/reviews_dataset_{subset}.json'
        with open(file_name) as f:
            data = json.load(f)

        replacement_words = ['instead', 'substitute', 'in place of', 'replace']
        self.sentences = []
        self.labels = []
        for d in data:
            if any(word in d['text'] for word in replacement_words):  # only accept certain sentences
                text = d['text']
                self.sentences.append(text)
                self.labels.append(tuple(d['label']))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], torch.tensor(self.labels[idx], dtype=torch.float32)
