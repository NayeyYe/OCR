# test/test_utils.py
import os
import unittest
import numpy as np
import torch
from src.utils.metrics import ClassificationMetrics
from src.utils.visualization import plot_confusion_matrix


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.num_classes = 5
        self.metrics = ClassificationMetrics(self.num_classes)

        # 生成测试数据
        self.outputs = torch.randn(10, self.num_classes)
        self.targets = torch.randint(0, self.num_classes, (10,))

    def test_metrics_update(self):
        """测试指标更新"""
        self.metrics.update(self.outputs, self.targets)
        self.assertEqual(self.metrics.confusion.sum().item(), 10)

    def test_compute_accuracy(self):
        """测试准确率计算"""
        # 有效数据测试
        self.metrics.confusion = torch.eye(5)
        metrics = self.metrics.compute()
        self.assertAlmostEqual(metrics['accuracy'], 1.0, delta=1e-6)

        # 空数据测试
        empty_metrics = ClassificationMetrics(5)
        metrics = empty_metrics.compute()
        self.assertEqual(metrics['accuracy'], 0.0)


class TestVisualization(unittest.TestCase):
    def test_confusion_matrix_plot(self):
        """测试混淆矩阵生成"""
        # 测试正常情况（无警告）
        cm = np.eye(20, dtype=int) * 10
        class_names = [f"class_{i}" for i in range(20)]

        # 不使用日志断言（正常情况不应触发警告）
        plot_confusion_matrix(cm, class_names, "test_cm.png")
        self.assertTrue(os.path.exists("test_cm.png"))
        os.remove("test_cm.png")

        # 测试空数据情况（有警告）
        with self.assertLogs(level='WARNING') as cm_log:
            plot_confusion_matrix(
                np.zeros((5, 5), dtype=int),
                ["c1", "c2", "c3", "c4", "c5"],
                "empty.png"
            )
        # 验证日志内容
        self.assertIn("所有类别的支持数均为零", cm_log.output[0])


if __name__ == '__main__':
    unittest.main()
