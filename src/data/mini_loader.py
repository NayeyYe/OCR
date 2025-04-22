import os
import random
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


class MiniCharDataset(Dataset):
    """小规模字符数据集"""

    def __init__(self, root_dir, transform=None, max_classes=100, max_samples=10):
        self.classes = sorted(os.listdir(root_dir))[:max_classes]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        # 随机采样每个类别的样本
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)]
            selected = random.sample(images, min(len(images), max_samples))
            self.samples.extend([(img, self.class_to_idx[cls]) for img in selected])

        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        return self.transform(img), label


def create_mini_loaders(config):
    # 修正后的数据增强
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 明确转换为单通道
        transforms.RandomRotation(5),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 验证集同样处理
        transforms.ToTensor()
    ])
    # 创建数据集
    full_train = MiniCharDataset(
        config.train_dir,
        transform=train_transform,  # 使用修正后的transform
        max_classes=config.num_classes,
        max_samples=config.num_samples_per_class
    )
    # 随机划分训练/验证集
    total_size = len(full_train)
    train_size = int(0.8 * total_size)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_loader = DataLoader(
        Subset(full_train, indices[:train_size]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        Subset(full_train, indices[train_size:]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, val_loader
