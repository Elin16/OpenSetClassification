import torch
class MahalanobisClassifier:

    def fit(self, X, y):

        # normalize embeddings
        X = X / torch.norm(X, dim=1, keepdim=True)

        self.classes = torch.unique(y)

        self.class_means = {}

        for c in self.classes:
            self.class_means[int(c)] = X[y == c].mean(dim=0)

        # global covariance
        cov = torch.cov(X.T)

        # regularization (critical)
        eps = 1e-3
        cov = cov + eps * torch.eye(cov.shape[0])

        self.inv_cov = torch.inverse(cov)

    def score(self, X):

        X = X / torch.norm(X, dim=1, keepdim=True)

        preds = []
        scores = []

        for x in X:

            best_dist = float("inf")
            best_class = None

            for c, mean in self.class_means.items():

                diff = x - mean

                dist = torch.matmul(
                    torch.matmul(diff, self.inv_cov),
                    diff
                )

                dist = dist.item()

                if dist < best_dist:
                    best_dist = dist
                    best_class = c

            preds.append(best_class)

            # negative distance as score
            scores.append(-best_dist)

        return torch.tensor(preds), torch.tensor(scores)