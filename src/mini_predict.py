# src/mini_predict.py
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.models.cnn import OCRCNN
from src.configs.mini_config import MiniConfig

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MiniPredictor:
    def __init__(self, model_path):
        self.cfg = MiniConfig()
        self.device = torch.device(self.cfg.device)
        self.model = OCRCNN(num_classes=self.cfg.num_classes).to(self.device)
        self._load_model(model_path)
        self.class_names = self._get_class_names()
        self.transform = self._build_transform()

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)

        # 多格式兼容处理
        state_dict = checkpoint.get('model_state',
                                    checkpoint.get('state_dict', checkpoint))

        # 去除DataParallel前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 非严格加载
        load_info = self.model.load_state_dict(state_dict, strict=False)
        print(f"成功加载参数: {len(state_dict) - len(load_info.missing_keys)}/{len(state_dict)}")

    def _get_class_names(self):
        class_dirs = sorted(os.listdir(self.cfg.train_dir))[:self.cfg.num_classes]
        return [d.split('_')[-1] for d in class_dirs]

    def _build_transform(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict(self, image_path, topk=5):
        img = Image.open(image_path).convert('L')
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, topk)

        return [(self.class_names[i], p.item())
                for p, i in zip(top_probs[0], top_indices[0])]

    def visualize(self, image_path, topk=5):
        predictions = self.predict(image_path, topk)
        plt.figure(figsize=(12, 6))

        # 显示原图
        plt.subplot(1, 2, 1)
        img = Image.open(image_path)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        # 显示预测
        plt.subplot(1, 2, 2)
        names = [f"{name}\n{prob * 100:.1f}%"
                 for name, prob in predictions]
        probs = [prob for _, prob in predictions]

        plt.barh(range(topk), probs[::-1], color='skyblue')
        plt.yticks(range(topk), names[::-1])
        plt.xlabel('Probability')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    predictor = MiniPredictor("../models/mini_models/mini_model_20250420_1844.pth")
    predictor.visualize("../data/test.png")
