import torch
import numpy as np
from embeddings.embedding_loader import load_embedding
from utils.split import split_known_unknown_classes, split_train_test
from metrics.evaluation import evaluate

from classifiers.cosine import CosineClassifier
from classifiers.euclidean import EuclideanClassifier
from classifiers.mahalanobis import MahalanobisClassifier
from classifiers.linear import LinearClassifier
from classifiers.energy import EnergyClassifier
from classifiers.openmax import OpenMaxClassifier

from rejection_methods.global_threshold import GlobalThreshold
from rejection_methods.class_threshold import ClassThreshold
from rejection_methods.class_percentile_threshold import ClassThresholdPercentile
from experiment.visualization_compare import visualize_classifier_prototypes

CLASSIFIERS = {
    "cosine": CosineClassifier,
    "euclidean": EuclideanClassifier,
    "mahalanobis": MahalanobisClassifier,
    "linear": LinearClassifier,
    "energy": EnergyClassifier,
    "openmax": OpenMaxClassifier
}
REJECTION_METHODS = {
    "global_threshold": GlobalThreshold,
    "class_threshold": ClassThreshold,
    "class_percentile": ClassThresholdPercentile
}

def run_experiment(
        embedding,
        rejection_method,
        classifier_name,
        num_known,
        seed,
        test_fold):

    X, y, folds = load_embedding(embedding)
    
    known_classes, unknown_classes = split_known_unknown_classes(
        y,
        num_known,
        seed
    )

    X_train, y_train, X_test, y_test = split_train_test(
        X,
        y,
        folds,
        test_fold
    )

    train_mask = torch.isin(y_train, known_classes)

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    model = CLASSIFIERS[classifier_name]()

    model.fit(X_train, y_train)

    pred_train, score_train = model.score(X_train)

    rejector = REJECTION_METHODS[rejection_method]()

    rejector.fit(
        pred_train,
        score_train,
        y_train
    )

    pred_test, score_test = model.score(X_test)

    pred_test = rejector.apply(
        pred_test,
        score_test
    )

    threshold = rejector.get_threshold()
 
    print("threshold:", rejector.get_threshold())
    print("train score min:", score_train.min())
    print("train score max:", score_train.max())
    print("test score min:", score_test.min())
    print("test score max:", score_test.max())
    print("")
    metrics = evaluate(
        pred_test,
        score_test,
        y_test,
        known_classes
    )

                        
    if classifier_name in ["cosine", "euclidean", "mahalanobis"]:
        visualize_classifier_prototypes(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            known_classes=known_classes,
            rejector=rejector,
            classifier_types=[classifier_name],
            method="tsne"  # 或 "umap"
        )
    
    return threshold, metrics