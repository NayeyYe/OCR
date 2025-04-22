import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import cm
import torch

def save_training_curve(history, save_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))
    # 训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 验证准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Acc', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # 保存图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, top_n=20):
    """绘制混淆矩阵热力图（显示前top_n个类别）"""
    # 类型强制转换
    cm = cm.astype(np.int64)

    # 有效性检查
    support = cm.sum(axis=1)
    valid_indices = np.where(support > 0)[0]

    if len(valid_indices) == 0:
        logging.warning("所有类别的支持数均为零，跳过混淆矩阵绘制")
        return

    top_indices = valid_indices[np.argsort(-support[valid_indices])[:top_n]]
    cm_top = cm[np.ix_(top_indices, top_indices)]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_top,
        annot=True,
        fmt="d",  # 确保使用整数格式
        xticklabels=np.array(class_names)[top_indices],
        yticklabels=np.array(class_names)[top_indices],
        cmap='Blues',
        annot_kws={"size": 8}
    )
    plt.title(f'Confusion Matrix (Top {len(top_indices)} Classes)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_embeddings(features, labels, num_classes=50, save_path='embeddings.png'):
    """使用t-SNE可视化特征空间"""
    # 随机采样
    indices = np.random.choice(len(features), 1000, replace=False)
    features = features[indices]
    labels = labels[indices]
    # 降维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(features)
    # 可视化
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=labels, cmap=cm.get_cmap('tab20', num_classes),
        alpha=0.6, edgecolors='w', linewidths=0.5
    )
    # 添加颜色条
    cbar = plt.colorbar(scatter, boundaries=np.arange(num_classes+1)-0.5)
    cbar.set_ticks(np.arange(num_classes))
    cbar.set_ticklabels(np.arange(num_classes))
    plt.title('t-SNE Feature Visualization')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
def plot_sample_predictions(images, true_labels, pred_labels, class_names, save_dir, num_samples=16):
    """可视化预测样本对比"""
    plt.figure(figsize=(16, 16))
    for i in range(num_samples):
        plt.subplot(4, 4, i+1)
        image = images[i].permute(1, 2, 0).cpu().numpy()
        plt.imshow(image, cmap='gray')
        true_name = class_names[true_labels[i]]
        pred_name = class_names[pred_labels[i]]
        color = 'green' if true_name == pred_name else 'red'
        plt.title(f"True: {true_name}\nPred: {pred_name}", color=color)
        plt.axis('off')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'prediction_samples.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# 在src/utils/visualization.py中添加以下新函数
def advanced_visualization(trainer, config):
    """动态可视化增强系统"""

    class VisualMonitor:
        def __init__(self):
            self.metrics_history = {
                'train_loss': [],
                'val_acc': [],
                'learning_rate': []
            }
            self.batch_loss = []
            # 新增训练步骤劫持
            self._original_training_loop = None

        def on_epoch_end(self, epoch, model, optimizer):
            """每个epoch结束时触发"""
            # 收集学习率
            lr = optimizer.param_groups[0]['lr']
            self.metrics_history['learning_rate'].append(lr)

            # 生成动态训练曲线
            self._plot_dynamic_curves(epoch, config.figures_dir)

            # 保存特征可视化
            if epoch % 5 == 0:
                self._visualize_features(model, config, epoch)

        def on_batch_end(self, loss):
            """每个batch结束时触发"""
            self.batch_loss.append(loss)

        def _plot_dynamic_curves(self, epoch, save_dir):
            """动态多指标曲线"""
            # 添加空数据保护
            if not self.batch_loss:
                print(f"Epoch {epoch}跳过动态曲线绘制 - 无批次损失数据")
                return
            plt.figure(figsize=(15, 5))

            # 训练损失（含平滑曲线）
            plt.subplot(1, 3, 1)
            window_size = max(1, len(self.batch_loss) // 20)
            smoothed = np.convolve(self.batch_loss,
                                   np.ones(window_size) / window_size,
                                   mode='valid')
            plt.plot(self.batch_loss, alpha=0.2, label='Raw')
            plt.plot(smoothed, label=f'Smooth (w={window_size})')
            plt.title('Batch Loss')
            plt.legend()

            # 验证准确率
            plt.subplot(1, 3, 2)
            plt.plot(self.metrics_history['val_acc'], marker='o')
            plt.title('Validation Accuracy')

            # 学习率变化
            plt.subplot(1, 3, 3)
            plt.plot(self.metrics_history['learning_rate'])
            plt.title('Learning Rate Schedule')

            plt.tight_layout()
            plt.savefig(f"{save_dir}/dynamic_metrics_epoch{epoch}.png")
            plt.close()

        def _visualize_features(self, model, config, epoch):
            """中间层特征可视化"""
            sample = torch.randn(1, 1, 256, 256).to(config.device)
            features = model.get_feature_maps(sample)

            plt.figure(figsize=(15, 10))
            for i, feat in enumerate(features[:4]):
                plt.subplot(2, 2, i + 1)
                channel_mean = feat.mean(0)[0].cpu().numpy()
                plt.imshow(channel_mean, cmap='viridis')
                plt.title(f'Conv Block {i + 1} Feature Map')
                plt.axis('off')

            plt.savefig(f"{config.figures_dir}/features_epoch{epoch}.png")
            plt.close()

        def _hijack_training_loop(self, trainer):
            """劫持训练循环以捕获批次损失"""
            original_train_epoch = trainer.train_epoch

            def wrapped_train_epoch(*args, **kwargs):
                for batch in trainer.train_loader:
                    loss = trainer._process_batch(batch)  # 假设存在_process_batch方法
                    self.on_batch_end(loss.item())
                return original_train_epoch(*args, **kwargs)

            trainer.train_epoch = wrapped_train_epoch

    monitor = VisualMonitor()
    monitor._hijack_training_loop(trainer)

    # 包装原有训练方法
    original_train = trainer.train

    def wrapped_train():
        for epoch in range(config.epochs):
            original_train()  # 保持原有训练流程
            monitor.on_epoch_end(epoch, trainer.model, trainer.optimizer)
        return monitor.metrics_history

    trainer.train = wrapped_train
    return monitor
