import torch
class EuclideanClassifier:

    def fit(self, X, y):

        self.classes = torch.unique(y)
        self.class_means = {}

        for c in self.classes:
            mean = X[y == c].mean(dim=0)
            self.class_means[int(c)] = mean

    def score(self, X):

        preds = []
        scores = []

        for x in X:

            best_dist = float("inf")
            best_class = None

            for c, mean in self.class_means.items():

                d = torch.norm(x - mean)

                if d < best_dist:
                    best_dist = d
                    best_class = c

            preds.append(best_class)

            # convert distance to score
            scores.append(-best_dist)

        return torch.tensor(preds), torch.tensor(scores)
    