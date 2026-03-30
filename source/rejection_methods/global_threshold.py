import numpy as np
import torch

UNKNOWN_LABEL = -1


class GlobalThreshold:

    def __init__(self, num_steps=200):
        self.num_steps = num_steps
        self.threshold = None

    """
    全局阈值（使用第一版未知类检测逻辑）
    """
    def __init__(self, num_steps=200):
        self.num_steps = num_steps
        self.threshold = None

    def fit(self, train_preds, train_scores, y_train, known_classes=None):
        """
        train_preds: 分类器预测结果 (tensor)
        train_scores: 分类器对应分数 (tensor)
        y_train: 训练集真实标签 (tensor)
        known_classes: 已知类 (tensor)
        """
        # 训练集全为已知类的情况
        if known_classes is None:
            known_classes = torch.unique(y_train)

        scores = train_scores.detach().cpu().numpy()
        labels = y_train.detach().cpu().numpy()

        # 已知/未知掩码
        known_mask = np.isin(labels, known_classes.detach().cpu().numpy())
        true_unknown = ~known_mask

        score_min = scores.min()
        score_max = scores.max()

        best_acc = 0
        best_t = score_min

        # 遍历阈值寻找使未知类检测准确率最大的 t
        for t in np.linspace(score_min, score_max, self.num_steps):
            pred_unknown = scores < t
            acc = np.mean(pred_unknown == true_unknown)
            if acc > best_acc:
                best_acc = acc
                best_t = t

        self.threshold = best_t

    def apply(self, pred_classes, scores):

        preds = pred_classes.detach().cpu().numpy().copy()
        scores = scores.detach().cpu().numpy().copy()

        preds[scores < self.threshold] = UNKNOWN_LABEL

        return torch.tensor(preds)
    
    def get_threshold(self):
        return self.threshold