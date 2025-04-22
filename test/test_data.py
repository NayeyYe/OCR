import unittest
import torch
import os
from src.data.loader import OCRDataset, create_loaders, get_transforms
import tempfile
import shutil
from PIL import Image
import numpy as np


class TestOCRDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """创建临时测试数据集"""
        cls.test_dir = tempfile.mkdtemp()
        cls.transform = get_transforms()  # 获取标准转换流程
        cls.create_test_structure(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        """清理临时文件"""
        shutil.rmtree(cls.test_dir)

    @classmethod
    def create_test_structure(cls, root_dir):
        """创建符合项目规范的测试数据"""
        # 创建3个测试类别
        for i in range(3):
            class_dir = os.path.join(root_dir, f"class_{i}")
            os.makedirs(class_dir, exist_ok=True)

            # 生成256x256灰度图像
            for j in range(5):
                # 创建白底黑字模拟图像
                arr = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
                img = Image.fromarray(arr, mode='L')
                img.save(os.path.join(class_dir, f"test_{j}.png"))

    def test_sample_shape(self):
        """验证数据转换后的形状"""
        dataset = OCRDataset(
            root_dir=self.test_dir,
            transform=self.transform  # 关键修正：应用数据转换
        )
        img_tensor, label = dataset[0]

        # 验证张量形状和数值范围
        self.assertEqual(img_tensor.shape, (1, 256, 256))  # 通道×高×宽
        self.assertTrue(torch.all(img_tensor >= -1.0))
        self.assertTrue(torch.all(img_tensor <= 1.0))


if __name__ == '__main__':
    unittest.main()
