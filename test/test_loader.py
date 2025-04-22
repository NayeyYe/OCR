# 临时测试脚本 test_loader.py
from src.configs.config import cfg
from data.loader import OCRDataset, create_loaders

if __name__ == "__main__":
    train_loader, val_loader = create_loaders(cfg)
    sample, label = next(iter(train_loader))
    print(f"数据形状: {sample.shape}")
    print(f"标签示例: {label[:5]}")
