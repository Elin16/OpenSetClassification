import torch


def split_known_unknown_classes(labels, num_known, seed=0):

    torch.manual_seed(seed)

    classes = torch.unique(labels)

    perm = torch.randperm(len(classes))

    known_classes = classes[perm[:num_known]]
    unknown_classes = classes[perm[num_known:]]

    return known_classes, unknown_classes


def split_train_test(X, y, folds, test_fold):

    train_mask = folds != test_fold
    test_mask = folds == test_fold

    return (
        X[train_mask],
        y[train_mask],
        X[test_mask],
        y[test_mask],
    )