# src/train.py
import torch
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch import nn, optim
from data.loader import create_loaders
from models.cnn import OCRCNN
from configs.config import cfg
from utils.logger import setup_logger
from utils.metrics import ClassificationMetrics
from utils.visualization import save_training_curve
from src.utils.visualization import advanced_visualization


def main():
    logger = setup_logger(cfg)
    scaler = GradScaler()
    metrics = ClassificationMetrics(cfg.num_classes)

    try:
        # 数据准备
        train_loader, val_loader = create_loaders(cfg)
        logger.info(f"训练集: {len(train_loader.dataset)} 验证集: {len(val_loader.dataset)}")

        # 模型初始化
        model = OCRCNN(cfg.num_classes).to(cfg.device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr,
            steps_per_epoch=len(train_loader), epochs=cfg.epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        history = {'train_loss': [], 'val_acc': []}

        trainer = type('', (), {'model': model, 'optimizer': optimizer, 'train': lambda: ..., 'train_epoch': cfg.epochs})
        advanced_visualization(trainer, cfg)

        # 初始损失验证
        with torch.no_grad():
            dummy_input = torch.randn(2, 1, 256, 256).to(cfg.device)
            dummy_loss = criterion(model(dummy_input),
                                   torch.randint(0, cfg.num_classes, (2,)).to(cfg.device))
            logger.info(f"初始损失验证: {dummy_loss.item():.2f} (预期≈{torch.log(torch.tensor(cfg.num_classes)):.2f})")

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0
            progress = tqdm(train_loader,
                            desc=f'Epoch {epoch + 1}/{cfg.epochs}',
                            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

            for images, labels in progress:
                images = images.to(cfg.device, non_blocking=True)
                labels = labels.to(cfg.device, non_blocking=True)

                # 混合精度训练
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

            # 验证
            val_acc, conf_matrix = evaluate(model, val_loader, cfg, metrics)
            history['train_loss'].append(epoch_loss / len(train_loader))
            history['val_acc'].append(val_acc)

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({'model_state': model.state_dict()},
                           cfg.best_model_path)
                logger.info(f"新最佳准确率: {val_acc * 100:.2f}%")

        save_training_curve(history, cfg.figures_dir)
        logger.info(f"训练完成 | 最佳准确率: {best_acc * 100:.2f}%")

    except Exception as e:
        logger.error(f"训练异常: {str(e)}")
        raise


def evaluate(model, loader, cfg, metrics):
    model.eval()
    metrics.reset()

    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for images, labels in tqdm(loader, desc='Validating'):
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            outputs = model(images)
            metrics.update(outputs, labels)

    results = metrics.compute()
    return results['accuracy'], results['confusion_matrix']


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
