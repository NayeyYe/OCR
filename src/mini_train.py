# src/mini_train.py
import os
import time
import torch
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch import nn, optim
from src.data.mini_loader import create_mini_loaders
from src.models.cnn import OCRCNN
from src.configs.mini_config import MiniConfig
from src.utils.visualization import advanced_visualization

class MiniTrainer:
    def __init__(self):
        self.cfg = MiniConfig()
        self.model = OCRCNN(self.cfg.num_classes).to(self.cfg.device)
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.cfg.lr,
                                     weight_decay=self.cfg.weight_decay)
        self.scaler = GradScaler()
        self.criterion = nn.CrossEntropyLoss()
        self.trainer = type('', (), {'model': self.model, 'optimizer': self.optimizer, 'train': lambda: ..., 'train_epoch': self.cfg.epochs})
        advanced_visualization(self.trainer, self.cfg)
    def train(self):
        train_loader, val_loader = create_mini_loaders(self.cfg)
        best_acc = 0.0

        for epoch in range(self.cfg.epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0.0
            progress = tqdm(train_loader,
                            desc=f"Mini Epoch {epoch + 1}/{self.cfg.epochs}",
                            colour='GREEN')

            for images, labels in progress:
                images = images.to(self.cfg.device, non_blocking=True)
                labels = labels.to(self.cfg.device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

            # 快速验证
            val_acc = self._evaluate(val_loader)
            print(f"Epoch {epoch + 1} | Val Acc: {val_acc * 100:.2f}%")

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                self._save_model()

    def _evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            for images, labels in loader:
                images = images.to(self.cfg.device)
                labels = labels.to(self.cfg.device)

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def _save_model(self):
        os.makedirs(self.cfg.model_save_dir, exist_ok=True)
        save_path = os.path.join(
            self.cfg.model_save_dir,
            f"mini_model_{time.strftime('%Y%m%d_%H%M')}.pth"
        )
        torch.save({
            'model_state': self.model.state_dict(),
            'num_classes': self.cfg.num_classes,
            'input_size': (1, 256, 256)
        }, save_path)
        print(f"\n模型已保存至: {save_path}")


if __name__ == "__main__":
    trainer = MiniTrainer()
    start = time.time()
    trainer.train()
    print(f"\n总训练时间: {time.time() - start:.1f}s")
    print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1e6:.1f}MB")
