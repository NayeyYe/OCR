from utils.logger import enable_logging
import os
import shutil


def move_folder(src_folder, dst_parent, overwrite=True):
    """
    将源文件夹移动到目标父目录下，并保持原文件夹名称

    :param src_folder: 要移动的源文件夹路径
    :param dst_parent: 目标父目录路径
    :param overwrite: 是否覆盖已存在的目标文件夹（默认True）
    """
    try:
        # 验证源文件夹
        if not os.path.exists(src_folder):
            print(f"[跳过] 源文件夹不存在: {src_folder}")
            return False
        if not os.path.isdir(src_folder):
            raise ValueError(f"源路径不是文件夹: {src_folder}")

        # 创建目标父目录（如果不存在）
        os.makedirs(dst_parent, exist_ok=True)

        # 构造完整目标路径
        folder_name = os.path.basename(src_folder)
        dst_path = os.path.join(dst_parent, folder_name)

        # 处理目标已存在的情况
        if os.path.exists(dst_path):
            if not overwrite:
                raise FileExistsError(f"目标文件夹已存在: {dst_path}")

            print(f"正在清理已存在的目标文件夹: {dst_path}")
            if os.path.isdir(dst_path):
                shutil.rmtree(dst_path)  # 递归删除目录
            else:
                os.remove(dst_path)  # 删除文件

        # 执行移动操作
        shutil.move(src_folder, dst_parent)
        print(f"成功移动文件夹: {src_folder} -> {dst_path}")
        return True

    except Exception as e:
        print(f"× 操作异常: {str(e)}")
        return False


if __name__ == "__main__":
    enable_logging()
    sub_files = ["raw", "processed"]
    sub_sub_files = ["Test", "Train"]
    # 使用示例
    for sub_file in sub_files:
        for sub_sub_file in sub_sub_files:
            source = os.path.join("..", "..", "data", sub_file, sub_sub_file)  # 替换为实际源路径
            destination = os.path.join("..", "..", "data", "letter", sub_file, sub_sub_file)  # 替换为实际目标父目录

            # 执行移动操作（overwrite=False 为安全模式）
            move_folder(source, destination, overwrite=True)
