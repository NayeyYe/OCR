# src/utils/metrics.py
import torch
from sklearn.metrics import classification_report


class ClassificationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.confusion = torch.zeros((self.num_classes, self.num_classes),
                                     dtype=torch.int64)

    def update(self, outputs, targets):
        _, preds = torch.max(outputs, 1)
        self.y_true.extend(targets.cpu().tolist())
        self.y_pred.extend(preds.cpu().tolist())

        for t, p in zip(targets.view(-1), preds.view(-1)):
            self.confusion[t.long(), p.long()] += 1

    def compute(self):
        total_samples = self.confusion.sum().item()

        # 防止空数据导致除零错误
        accuracy = self.confusion.diag().sum().item() / total_samples if total_samples > 0 else 0.0

        # 防止混淆矩阵全零时分类报告错误
        report = classification_report(
            self.y_true, self.y_pred,
            output_dict=True,
            zero_division=0
        ) if total_samples > 0 else {}

        return {
            'accuracy': accuracy,
            'confusion_matrix': self.confusion.numpy(),
            'report': report
        }

