# src/data/loader.py
import albumentations as A
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from PIL import Image
import cv2

class OCRDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_classes=None):
        self.root = Path(root_dir)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])[:max_classes] \
        # 新增样本限制逻辑
        self.sample_limit = 100 if "Train" in str(self.root) else 10
        self.samples = self._load_samples()
        self.transform = transform

    def _load_samples(self):
        """按文件名数字顺序加载前N个样本"""
        samples = []
        for cls_idx, cls in enumerate(self.classes):
            cls_dir = self.root / cls
            # 按文件名数字排序（假设文件名为1.png,2.png,...）
            files = sorted(cls_dir.glob("*.png"),
                           key=lambda x: int(x.stem))
            # 截取前N个样本
            files = files[:self.sample_limit]
            samples.extend([(fpath, cls_idx) for fpath in files])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert('L'))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img.float(), label


# src/data/loader.py
def get_transforms(train=True):
    base = [
        A.Resize(256, 256),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ]

    if train:
        return A.Compose([
            # 替换前：A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
            A.Affine(translate_percent=(0.0, 0.1), scale=(0.9, 1.1), rotate=(-15, 15), interpolation=cv2.INTER_LINEAR),
            A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            *base
        ])
    return A.Compose(base)


def create_loaders(config):
    train_set = OCRDataset(config.train_dir,
                           transform=get_transforms(True),
                           max_classes=config.num_classes)
    val_set = OCRDataset(config.val_dir,
                         transform=get_transforms(False),
                         max_classes=config.num_classes)

    return (
        DataLoader(train_set, batch_size=config.batch_size,
                   shuffle=True, num_workers=config.num_workers,
                   pin_memory=True),
        DataLoader(val_set, batch_size=config.batch_size,
                   shuffle=False, num_workers=config.num_workers,
                   pin_memory=True)
    )
