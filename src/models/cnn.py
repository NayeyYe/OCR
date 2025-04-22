import torch
import torch.nn as nn
import torch.nn.init as init


class OCRCNN(nn.Module):
    """高精度OCR卷积神经网络"""

    def __init__(self, num_classes=7304):
        super(OCRCNN, self).__init__()

        # 特征提取模块
        self.features = nn.Sequential(
            # 输入: [1, 256, 256]
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 输出: [32, 128, 128]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 输出: [64, 64, 64]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 输出: [128, 32, 32]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 输出: [256, 16, 16]
        )

        # 分类模块
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        """用于特征可视化的中间层输出"""
        features = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x.detach().cpu())
        return features


if __name__ == '__main__':
    cnn = OCRCNN()
    print(cnn)