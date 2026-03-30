import numpy as np
import torch

UNKNOWN_LABEL = -1

class ClassQuantileThreshold:
    """
    Class-wise rejection method based on score quantiles.
    Computes thresholds for each known class using the specified quantile.
    """

    def __init__(self, quantile=0.05):
        """
        Args:
            quantile (float): the lower quantile of scores to use as threshold.
                              e.g., 0.05 means 5% quantile, scores below this are rejected.
        """
        self.quantile = quantile
        self.thresholds = {}

    def fit(self, train_preds, train_scores, y_train):
        """
        Fit thresholds per class based on training scores.
        For each known class, compute the lower quantile score as threshold.

        Args:
            train_preds (torch.Tensor): predicted class labels for training set (not used here)
            train_scores (torch.Tensor): confidence scores for training set
            y_train (torch.Tensor): true labels of training set
        """
        scores = train_scores.detach().cpu().numpy()
        labels = y_train.detach().cpu().numpy()

        classes = np.unique(labels)
        thresholds = {}

        for c in classes:
            class_scores = scores[labels == c]
            if len(class_scores) == 0:
                continue
            # Use quantile as threshold
            thresholds[c] = np.quantile(class_scores, self.quantile)

        self.thresholds = thresholds

    def apply(self, pred_classes, scores):
        """
        Apply class-wise thresholds to predictions.

        Args:
            pred_classes (torch.Tensor): predicted class labels
            scores (torch.Tensor): predicted confidence scores

        Returns:
            torch.Tensor: predicted labels with UNKNOWN_LABEL for rejected samples
        """
        preds = pred_classes.detach().cpu().numpy().copy()
        scores_np = scores.detach().cpu().numpy()

        for i in range(len(preds)):
            c = preds[i]
            if c not in self.thresholds:
                preds[i] = UNKNOWN_LABEL
                continue
            if scores_np[i] < self.thresholds[c]:
                preds[i] = UNKNOWN_LABEL

        # Return tensor on same device as input
        return torch.tensor(preds, device=pred_classes.device)

    def get_threshold(self):
        """
        Return the computed thresholds per class.
        """
        return self.thresholds