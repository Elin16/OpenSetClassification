import numpy as np
import torch


def compute_oscr(scores, pred_classes, true_labels, known_classes):

    scores = scores.numpy()
    preds = pred_classes.numpy()
    labels = true_labels.numpy()

    known_mask = np.isin(labels, known_classes.numpy())
    unknown_mask = ~known_mask

    thresholds = np.sort(scores)

    CCR = []
    FPR = []

    for t in thresholds:

        accept = scores >= t

        correct_known = (
            accept
            & known_mask
            & (preds == labels)
        )

        ccr = correct_known.sum() / known_mask.sum()

        false_pos = (
            accept
            & unknown_mask
        )

        fpr = false_pos.sum() / unknown_mask.sum()

        CCR.append(ccr)
        FPR.append(fpr)

    CCR = np.array(CCR)
    FPR = np.array(FPR)

    idx = np.argsort(FPR)

    return np.trapz(CCR[idx], FPR[idx])