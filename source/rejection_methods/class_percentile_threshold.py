import numpy as np
import torch

UNKNOWN_LABEL = -1


class ClassThresholdPercentile:

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
            # 用真实标签筛选该类样本
            mask = labels == c
            class_scores = scores[mask]

            if len(class_scores) == 0:
                continue

            # ===== 新逻辑：使用95%置信区间 =====
            # 下5%分位数作为threshold（过滤掉最低的异常值）
            t = np.percentile(class_scores, 5)

            thresholds[c] = t

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