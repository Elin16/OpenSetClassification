import torch
import torch.nn as nn
import numpy as np

UNKNOWN_LABEL = -1

class OpenMaxClassifier:

    def __init__(self):
        self.model = None
        self.mavs = None
        self.y_map = None
        self.y_map_inv = None

    def fit(self, X, y):
        # 所有类
        classes = torch.unique(y)
        self.y_map = {int(c): i for i, c in enumerate(classes)}
        self.y_map_inv = {i: int(c) for i, c in enumerate(classes)}

        y_train = torch.tensor([self.y_map[int(i)] for i in y], dtype=torch.long)

        num_classes = len(classes)

        self.model = nn.Linear(X.shape[1], num_classes)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(200):
            logits = self.model(X)
            loss = criterion(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute MAV
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X).cpu().numpy()
            y_np = y_train.cpu().numpy()

            self.mavs = []
            for c in range(num_classes):
                cls_logits = logits[y_np == c]
                mav = cls_logits.mean(axis=0)
                self.mavs.append(mav)
            self.mavs = np.stack(self.mavs)

    def score(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X).cpu().numpy()

        preds = []
        scores = []

        for logit in logits:
            pred = np.argmax(logit)
            dist = np.linalg.norm(logit - self.mavs[pred])
            preds.append(self.y_map_inv[pred])
            scores.append(-dist)  # score越大越可能是已知类

        preds = torch.tensor(preds)
        scores = torch.tensor(scores)
        return preds, scores