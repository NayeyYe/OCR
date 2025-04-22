import unittest
import torch
import tempfile
from pathlib import Path
from src.models.cnn import OCRCNN
from src.configs.mini_config import mini_cfg


class TestOCRModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dummy_input = torch.randn(2, 1, 256, 256)
        cls.num_classes = mini_cfg.num_classes

    def test_model_io(self):
        """测试模型保存/加载"""
        model = OCRCNN(num_classes=10)
        model.eval()  # 新增：设置为评估模式

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pth"
            torch.save(model.state_dict(), path)

            loaded_model = OCRCNN(num_classes=10)
            loaded_model.eval()  # 新增：设置为评估模式
            loaded_model.load_state_dict(torch.load(path))

            # 验证输出一致性
            with torch.no_grad():
                out1 = model(self.dummy_input)
                out2 = loaded_model(self.dummy_input)
            self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    def test_mixed_precision(self):
        """测试混合精度支持"""
        model = OCRCNN(self.num_classes).cuda()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(self.dummy_input.cuda())
            self.assertEqual(outputs.dtype, torch.float16)


if __name__ == '__main__':
    unittest.main()
