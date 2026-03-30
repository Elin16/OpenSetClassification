# experiment/visualization_compare.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import os


def compute_prototypes(X, y, known_classes, classifier_type="euclidean"):

    prototypes = []

    for c in known_classes:

        mask = y == c
        proto = X[mask].mean(axis=0)

        if classifier_type in ["cosine", "mahalanobis"]:
            proto = proto / np.linalg.norm(proto)

        prototypes.append(proto)

    return np.stack(prototypes)


def visualize_classifier_prototypes(
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        known_classes=None,
        rejector=None,
        classifier_types=["cosine", "euclidean", "mahalanobis"],
        method="tsne",
        title_prefix="Embedding Visualization",
        embedding_name="clap",
        save_dir="results/fig"):

    os.makedirs(save_dir, exist_ok=True)

    if torch.is_tensor(X_train):
        X_train = X_train.cpu().numpy()
    if torch.is_tensor(y_train):
        y_train = y_train.cpu().numpy()
    if X_test is not None and torch.is_tensor(X_test):
        X_test = X_test.cpu().numpy()
    if y_test is not None and torch.is_tensor(y_test):
        y_test = y_test.cpu().numpy()

    num_known = len(known_classes)

    if y_test is not None:
        num_unknown = len(np.unique(y_test)) - num_known
    else:
        num_unknown = 0

    for clf_type in classifier_types:

        X_vis = X_train.copy()

        # normalization
        if clf_type in ["cosine", "mahalanobis"]:

            X_vis = X_vis / np.linalg.norm(X_vis, axis=1, keepdims=True)

            if X_test is not None:
                X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
            else:
                X_test_norm = None

        else:
            X_test_norm = X_test

        # compute prototypes
        prototypes = compute_prototypes(
            X_vis,
            y_train,
            known_classes,
            classifier_type=clf_type
        )

        # =========================
        # 统一读取 threshold
        # =========================

        thresholds = {}

        if rejector is not None:

            th = rejector.get_threshold()

            # class threshold
            if isinstance(th, dict):
                thresholds = th

            # global threshold
            else:
                for c in known_classes:
                    thresholds[c] = th

        # =========================
        # 生成 threshold 边界点
        # =========================

        proto_boundary_points = []

        for i, c in enumerate(known_classes):

            proto = prototypes[i]

            t = thresholds.get(c, None)

            if t is None:
                proto_boundary_points.append(proto)
                continue

            direction = np.random.randn(proto.shape[0])
            direction = direction / np.linalg.norm(direction)

            boundary_point = proto + direction * t

            proto_boundary_points.append(boundary_point)

        proto_boundary_points = np.stack(proto_boundary_points)

        # =========================
        # combine data
        # =========================

        all_X = X_vis

        if X_test_norm is not None:
            all_X = np.vstack([all_X, X_test_norm])

        all_X = np.vstack([all_X, prototypes])
        all_X = np.vstack([all_X, proto_boundary_points])

        # =========================
        # dimensionality reduction
        # =========================

        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, random_state=0)

        elif method.lower() == "umap":
            reducer = umap.UMAP(n_components=2, random_state=0)

        else:
            raise ValueError("method should be 'tsne' or 'umap'")

        all_2d = reducer.fit_transform(all_X)

        idx = 0

        X_2d = all_2d[idx:idx + len(X_vis)]
        idx += len(X_vis)

        if X_test_norm is not None:
            X_test_2d = all_2d[idx:idx + len(X_test_norm)]
            idx += len(X_test_norm)
        else:
            X_test_2d = None

        proto_2d = all_2d[idx:idx + len(prototypes)]
        idx += len(prototypes)

        proto_boundary_2d = all_2d[idx:idx + len(prototypes)]

        # =========================
        # 计算 2D radius
        # =========================

        radii = np.linalg.norm(
            proto_boundary_2d - proto_2d,
            axis=1
        )

        # =========================
        # plot
        # =========================

        plt.figure(figsize=(10, 7))

        # train samples
        for c in np.unique(y_train):

            mask = y_train == c

            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=20,
                alpha=0.6,
                label=f"train {c}"
            )

        # test samples
        if X_test_2d is not None:

            for c in np.unique(y_test):

                mask = y_test == c

                if c in known_classes:
                    marker = "x"
                    label = f"test known {c}"
                else:
                    marker = "^"
                    label = f"test unknown {c}"

                plt.scatter(
                    X_test_2d[mask, 0],
                    X_test_2d[mask, 1],
                    s=30,
                    alpha=0.7,
                    marker=marker,
                    label=label
                )

        # prototypes
        plt.scatter(
            proto_2d[:, 0],
            proto_2d[:, 1],
            c="black",
            marker="*",
            s=200,
            label="prototype"
        )

        # threshold circles
        for i in range(len(prototypes)):

            circle = plt.Circle(
                (proto_2d[i, 0], proto_2d[i, 1]),
                radii[i],
                color="red",
                fill=False,
                linestyle="--",
                linewidth=2,
                alpha=0.8
            )

            plt.gca().add_patch(circle)

        plt.title(
            f"{title_prefix}\n"
            f"{clf_type.upper()} + {method.upper()} | "
            f"known={num_known}, unknown={num_unknown}"
        )

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8
        )

        plt.tight_layout()

        filename = (
            f"{embedding_name}_"
            f"{clf_type}_"
            f"{method}_"
            f"known{num_known}_"
            f"unknown{num_unknown}.png"
        )

        save_path = os.path.join(save_dir, filename)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print("figure saved:", save_path)

        plt.close()