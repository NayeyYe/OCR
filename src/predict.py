# src/predict.py
import os
import torch
import json
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from src.models.cnn import OCRCNN
from src.configs.config import Config


class OCRPredictor:
    def __init__(self, model_path=None):
        self.cfg = Config()
        self.device = torch.device(self.cfg.device)

        # 模型初始化
        self.model = OCRCNN(num_classes=self.cfg.num_classes).to(self.device)
        self._load_model(model_path or self.cfg.best_model_path)

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 类别映射
        self.class_names = self._get_class_names()

    def _load_model(self, model_path):
        """加载训练好的模型"""
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    def _get_class_names(self):
        """获取类别名称列表"""
        train_dir = os.path.join(self.cfg.data_root, "processed/Train")
        return sorted(os.listdir(train_dir))[:self.cfg.num_classes]

    def predict_single(self, image_path, topk=5):
        """单张图像预测"""
        img = Image.open(image_path)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = probs.topk(topk, dim=1)

        return [
            (self.class_names[i], p.item())
            for p, i in zip(top_probs[0], top_indices[0])
        ]

    def predict_batch(self, image_dir, batch_size=32):
        """批量预测"""
        image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # 创建数据集
        dataset = [(path, self.transform(Image.open(path)))
                   for path in image_paths]
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            num_workers=self.cfg.num_workers
        )

        results = {}
        with torch.no_grad():
            for batch in tqdm(loader, desc="Processing"):
                paths, tensors = batch
                outputs = self.model(tensors.to(self.device))
                _, preds = torch.max(outputs, 1)

                for path, pred in zip(paths, preds):
                    results[path] = self.class_names[pred.item()]

        return results

    def generate_report(self, image_dir, save_path):
        """生成预测报告"""
        results = self.predict_batch(image_dir)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to {save_path}")


if __name__ == "__main__":
    # 使用示例
    predictor = OCRPredictor()

    # 单张预测
    test_img = "data/test.png"
    if os.path.exists(test_img):
        print(predictor.predict_single(test_img))

    # # 批量预测
    # predictor.generate_report(
    #     image_dir="data/raw/Test",
    #     save_path="reports/predictions.json"
    # )
