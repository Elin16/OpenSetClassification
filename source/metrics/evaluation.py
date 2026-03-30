import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .oscr import compute_oscr

UNKNOWN_LABEL = -1


def apply_threshold(pred_classes, scores, thresholds):

    pred_classes = pred_classes.copy()

    for i in range(len(pred_classes)):

        c = pred_classes[i]

        if c not in thresholds:
            pred_classes[i] = UNKNOWN_LABEL
            continue

        if scores[i] < thresholds[c]:
            pred_classes[i] = UNKNOWN_LABEL

    return pred_classes


def evaluate(pred_classes, scores, true_labels, known_classes):

    pred_classes = np.asarray(pred_classes)
    scores = np.asarray(scores)
    true_labels = np.asarray(true_labels)

    known_mask = np.isin(true_labels, known_classes.numpy())
    unknown_mask = ~known_mask

    # ===============================
    # Unknown detection accuracy
    # ===============================

    unknown_pred = pred_classes[unknown_mask]
    unknown_acc = np.mean(unknown_pred == UNKNOWN_LABEL)

    # ===============================
    # Closed-set accuracy
    # ===============================

    known_pred = pred_classes[known_mask]
    known_true = true_labels[known_mask]

    closed_acc = np.mean(known_pred == known_true)

    # ===============================
    # AUROC
    # ===============================
    binary_labels = (~known_mask).astype(int)

    auroc = roc_auc_score(binary_labels, -scores)

    # ===============================
    # OSCR
    # ===============================
    oscr = compute_oscr(
        torch.tensor(scores),
        torch.tensor(pred_classes),
        torch.tensor(true_labels),
        known_classes
    )

    return closed_acc, unknown_acc, auroc, oscr