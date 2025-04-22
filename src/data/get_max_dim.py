import os
from PIL import Image
import concurrent.futures
import time
from PIL import Image, ImageOps
from src.utils.logger import enable_logging

def get_image_dimension(file_path):
    """获取单个图像的最大维度（长或宽）"""
    try:
        with Image.open(file_path) as img:
            return max(img.size)  # 直接返回长宽中的最大值
    except Exception as e:
        print(f"处理文件失败 {file_path}: {str(e)}")
        return 0


def scan_directory(root_dir):
    """扫描指定目录下的所有PNG文件"""
    png_files = []
    # 使用os.scandir提高遍历效率
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            for sub_entry in os.scandir(entry.path):
                if sub_entry.is_file() and sub_entry.name.lower().endswith('.png'):
                    png_files.append(sub_entry.path)
    return png_files


def get_max_dimension():
    start_time = time.time()
    max_dimension = 0
    total_files = 0

    # 配置扫描路径
    base_dir = "..\\..\\data\\raw"
    target_dirs = ["Train"]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for sub_dir in target_dirs:
            current_dir = os.path.join(base_dir, sub_dir)
            print(f"正在扫描目录: {current_dir}")

            # 获取所有PNG文件路径
            png_files = scan_directory(current_dir)
            total_files += len(png_files)

            # 分批次处理以避免内存溢出
            batch_size = 10000
            for i in range(0, len(png_files), batch_size):
                batch = png_files[i:i + batch_size]
                results = executor.map(get_image_dimension, batch)

                for dim in results:
                    if dim > max_dimension:
                        max_dimension = dim
                        print(f"发现新最大尺寸: {max_dimension}px")

    print(f"\n扫描完成，共处理 {total_files} 个文件")
    print(f"最大维度: {max_dimension} 像素")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")
