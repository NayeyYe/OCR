import os
import torch


class MiniConfig:
    def __init__(self):
        # 硬件配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = os.cpu_count()-2 # 减少工作线程数

        # 路径配置
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.train_dir = os.path.join(self.root_dir, "data/processed/Train")
        self.model_save_dir = os.path.join(self.root_dir, "models/mini_models")
        self.reports_dir = os.path.join(self.root_dir, "reports")
        self.figures_dir = os.path.join(self.reports_dir, "mini_figures")

        # 训练参数
        self.epochs = 1  # 减少训练轮次
        self.batch_size = 32  # 减小批大小
        self.lr = 1e-4  # 降低学习率
        self.subset_ratio = 0.01  # 使用1%的训练数据
        self.num_samples_per_class = 100  # 每类最多取100个样本
        self.weight_decay = 1e-4

        # 模型参数
        self.num_classes = 100  # 只取前100个类别

        # 初始化目录
        self._create_dirs()

    def _create_dirs(self):
        os.makedirs(self.model_save_dir, exist_ok=True)


mini_cfg = MiniConfig()