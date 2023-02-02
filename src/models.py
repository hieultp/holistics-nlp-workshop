import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, n_features=128) -> None:
        super().__init__()
        self.n_features = n_features

        self.model = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=n_features),
            nn.ReLU(),
            nn.Linear(in_features=n_features, out_features=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        pred = self.model(x)
        return pred


class ClassificationModel(nn.Module):
    def __init__(self, n_features=128) -> None:
        super().__init__()
        self.n_features = n_features

        self.model = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=n_features),
            nn.ReLU(),
            nn.Linear(in_features=n_features, out_features=10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        pred = self.model(x)
        return pred
