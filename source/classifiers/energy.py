import torch
import torch.nn as nn
import torch.optim as optim


class EnergyClassifier:

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def fit(self, X, y):

        self.classes = torch.unique(y)

        self.y_map = {c.item(): i for i, c in enumerate(self.classes)}
        self.y_map_inv = {v: k for k, v in self.y_map.items()}

        y_train = torch.tensor([self.y_map[i.item()] for i in y])

        self.model = nn.Linear(X.shape[1], len(self.classes))

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        loss_fn = nn.CrossEntropyLoss()

        self.model.train()

        for _ in range(200):

            logits = self.model(X)

            loss = loss_fn(logits, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def score(self, X):

        self.model.eval()

        with torch.no_grad():

            logits = self.model(X)

            # prediction
            preds_idx = torch.argmax(logits, dim=1)

            preds = torch.tensor(
                [self.y_map_inv[i.item()] for i in preds_idx]
            )

            # ENERGY SCORE
            energy = -torch.logsumexp(
                logits / self.temperature,
                dim=1
            )

            # convert to "known score"
            scores = -energy

        return preds, scores