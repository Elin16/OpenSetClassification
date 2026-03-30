import numpy as np
import torch

UNKNOWN_LABEL = -1


class ClassThreshold:

    def __init__(self, num_steps=100):
        self.num_steps = num_steps
        self.thresholds = {}

    def fit(self, train_preds, train_scores, y_train):
        # 转为 numpy
        scores = train_scores.detach().cpu().numpy()
        labels = y_train.detach().cpu().numpy()

        classes = np.unique(labels)
        thresholds = {}

        for c in classes:
            # 这里用真实标签筛选该类样本
            mask = labels == c
            class_scores = scores[mask]

            score_min = class_scores.min()
            score_max = class_scores.max()

            best_acc = 0
            best_t = score_min

            # 遍历 threshold 值，找到训练集上能最好区分该类样本的阈值
            for t in np.linspace(score_min, score_max, self.num_steps):
                # 对训练集上该类样本，分数 >= t 的判为已知
                pred_known = class_scores >= t
                acc = np.mean(pred_known)  # 全部都是 True，acc 越高越好

                if acc > best_acc:
                    best_acc = acc
                    best_t = t

            thresholds[c] = best_t

        self.thresholds = thresholds
        
    def fit_old(self, train_preds, train_scores, y_train):

        preds = train_preds.detach().cpu().numpy()
        scores = train_scores.detach().cpu().numpy()
        labels = y_train.detach().cpu().numpy()

        classes = np.unique(labels)

        thresholds = {}

        for c in classes:

            mask = preds == c

            if mask.sum() == 0:
                continue

            class_scores = scores[mask]

            score_min = class_scores.min()
            score_max = class_scores.max()

            true_labels = labels[mask] == c

            best_acc = 0
            best_t = score_min

            for t in np.linspace(score_min, score_max, self.num_steps):

                pred_known = class_scores >= t

                acc = np.mean(pred_known == true_labels)

                if acc > best_acc:
                    best_acc = acc
                    best_t = t

            thresholds[c] = best_t

        self.thresholds = thresholds

    def apply(self, pred_classes, scores):

        preds = pred_classes.detach().cpu().numpy().copy()
        scores = scores.detach().cpu().numpy()

        for i in range(len(preds)):

            c = preds[i]

            if c not in self.thresholds:
                preds[i] = UNKNOWN_LABEL
                continue

            if scores[i] < self.thresholds[c]:
                preds[i] = UNKNOWN_LABEL

        return torch.tensor(preds)
    
    def get_threshold(self):
        return self.thresholds