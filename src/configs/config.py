# src/configs/config.py
import os
import torch


class BaseConfig:
    def __init__(self):
        # 硬件配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = os.cpu_count() - 2

        # 路径配置
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_dir = os.path.join(self.root_dir, "data")
        self.model_dir = os.path.join(self.root_dir, "models")

        # 数据参数
        self.img_size = 256
        self.num_classes = 7186


class FullConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # 训练参数
        self.batch_size = 100
        self.epochs = 100
        self.lr = 1e-3
        self.weight_decay = 1e-4

        # 路径
        self.train_dir = os.path.join(self.data_dir, "processed/Train")
        self.val_dir = os.path.join(self.data_dir, "processed/Test")
        self.model_save_dir = os.path.join(self.model_dir, "saved_models")
        self.best_model_path = os.path.join(self.model_save_dir, "best_model.pth")

        self.reports_dir = os.path.join(self.root_dir, "reports")
        self.figures_dir = os.path.join(self.reports_dir, "figures")

        self.limit_train_samples = True


# 实例化配置
cfg = FullConfig()
