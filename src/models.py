import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, n_features=128) -> None:
        super().__init__()
        self.n_features = n_features

        self.model = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        pred = self.model(x)
        pred = pred.clip(max=9)
        return pred


class ClassificationModel(nn.Module):
    def __init__(self, n_features=128) -> None:
        super().__init__()
        self.n_features = n_features

        self.model = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=10),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        pred = self.model(x)
        return pred


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_class=3, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded.mean(dim=1))
