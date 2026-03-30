import torch

class CosineClassifier:

    def fit(self, X, y):

        self.classes = torch.unique(y)

        self.class_means = {}

        for c in self.classes:
            mean = X[y == c].mean(dim=0)
            mean = mean / torch.norm(mean)
            self.class_means[int(c)] = mean

    def score(self, X):

        scores = []
        preds = []

        for x in X:

            x = x / torch.norm(x)

            best_score = -1
            best_class = None

            for c, mean in self.class_means.items():

                s = torch.dot(x, mean)

                if s > best_score:
                    best_score = s
                    best_class = c

            scores.append(best_score)
            preds.append(best_class)

        return torch.tensor(preds), torch.tensor(scores)