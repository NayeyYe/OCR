# file: src/data/preprocess.py
import os
from PIL import Image, ImageOps
import concurrent.futures
import time
import sys
from tqdm import tqdm  # 添加进度条支持
import queue


def resize_and_pad(img, target_size=256, fill_color=(255, 255, 255)):
    """
    保持长宽比调整大小并填充到正方形
    参数：
        img: PIL Image对象
        target_size: 目标尺寸（默认256）
        fill_color: 填充颜色（默认白色）
    返回：
        处理后的PIL Image对象
    """
    # 计算缩放比例
    ratio = min(target_size / img.size[0], target_size / img.size[1])
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))

    # 保持原始图像模式（L为灰度，RGB为彩色）
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 高质量缩放
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # 创建新画布并居中粘贴
    new_img = Image.new('RGB', (target_size, target_size), fill_color)
    new_img.paste(img, ((target_size - new_size[0]) // 2,
                        (target_size - new_size[1]) // 2))
    return new_img


def process_single_image(src_path, dst_path):
    """处理单个图像文件"""
    try:
        with Image.open(src_path) as img:
            processed = resize_and_pad(img)
            processed.save(dst_path)
            return True
    except Exception as e:
        sys.stderr.write(f"\n处理失败 {src_path}: {str(e)}\n")
        return False


def batch_processor(root_dir, output_dir):
    """优化后的批量处理函数"""
    success = 0
    failure = 0

    # 先收集所有待处理文件路径
    file_pairs = []
    for class_dir in os.listdir(root_dir):
        src_class_path = os.path.join(root_dir, class_dir)
        dst_class_path = os.path.join(output_dir, class_dir)

        if not os.path.isdir(src_class_path):
            continue

        os.makedirs(dst_class_path, exist_ok=True)

        for filename in os.listdir(src_class_path):
            if filename.lower().endswith('.png'):
                src = os.path.join(src_class_path, filename)
                dst = os.path.join(dst_class_path, filename)
                file_pairs.append((src, dst))
    # 动态批处理设置
    BATCH_SIZE = 5000  # 根据内存调整
    MAX_WORKERS = max(1, os.cpu_count() - 2)  # 保留2个核心给系统

    with tqdm(total=len(file_pairs), desc="处理进度") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            result_queue = queue.Queue()

            # 提交第一批任务
            for src, dst in file_pairs[:BATCH_SIZE]:
                futures.append(executor.submit(process_single_image, src, dst))

            # 流水线处理
            index = BATCH_SIZE
            while futures:
                try:
                    future = next(f for f in futures if f.done())
                    futures.remove(future)

                    if future.result():
                        result_queue.put(True)
                    else:
                        result_queue.put(False)

                    # 提交新任务保持队列满载
                    if index < len(file_pairs):
                        src, dst = file_pairs[index]
                        futures.append(executor.submit(process_single_image, src, dst))
                        index += 1

                    # 更新计数
                    while not result_queue.empty():
                        if result_queue.get():
                            success += 1
                        else:
                            failure += 1
                        pbar.update(1)

                except StopIteration:
                    time.sleep(0.1)  # 防止空转
    return success, failure


def data_process():
    """主处理函数"""
    print("开始数据预处理...")
    start_time = time.time()

    # 路径配置
    base_raw_dir = os.path.join("..", "..", "data", "raw")
    base_processed_dir = os.path.join("..", "..", "data", "processed")

    # 处理训练集和测试集
    for dataset_type in ["Train", "Test"]:
        print(f"\n正在处理 {dataset_type} 数据集:")
        raw_dir = os.path.join(base_raw_dir, dataset_type)
        processed_dir = os.path.join(base_processed_dir, dataset_type)

        success, failure = batch_processor(raw_dir, processed_dir)

        print(f"{dataset_type} 完成：")
        print(f"成功: {success} 文件")
        print(f"失败: {failure} 文件")

    print(f"\n总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    data_process()


