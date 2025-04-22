
```
OCR/
├── data/                  # 数据管理
│   ├── raw/              # 原始数据（按类别分目录）
|	|	└── Character     # 中文、英文、数字、符号全字符数据集
│   └── processed/        # 预处理后的标准格式数据
├── models/               # 模型管理
│   ├── saved_models/     # 训练完成的模型权重
│   └── __init__.py       # 空文件（标识Python包）
├── src/                  # 核心代码
│   ├── configs/          # 配置管理
│   │   └── settings.py   # 超参数/路径配置
│   ├── core/             # 核心逻辑
│   │   ├── train.py      # 训练主流程
│   │   └── evaluate.py   # 评估逻辑
│   ├── data/             # 数据处理
│   │   ├── loader.py     # 数据加载
│   │   └── preprocess.py # 数据预处理
│   ├── model/            # 模型定义
│   │   ├── cnn.py        # 基础CNN
│   │   └── resnet.py     # 迁移学习模型
│   └── utils/            # 工具包
│       ├── logger.py     # 训练日志
│       ├── evaluate     #  模型评价
│       └── visualization.py  # 结果可视化
├── reports/              # 输出结果
│   ├── metrics/          # 评估指标
│   └── figures/          # 训练曲线/混淆矩阵
└── tests/                # 基础测试
     ├── test_data.py     # 数据测试
     ├── test_model.py    # 模型测试
     └── test_utils.py    # 工具函数测试
 
```

