import os
import cv2
import shutil
from pathlib import Path

def convert_dataset_to_grayscale(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ 错误: 输入路径不存在 -> {input_dir}")
        return

    # 定义常见的图片格式后缀
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

    # 递归遍历输入目录下的所有文件
    for file_path in input_path.rglob('*'):
        if not file_path.is_file():
            continue

        # 计算相对路径，以保持输出目录的结构一致
        relative_path = file_path.relative_to(input_path)
        target_file_path = output_path / relative_path

        # 确保目标文件所在的父目录存在
        target_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_suffix = file_path.suffix.lower()

        # 如果是图片文件，转换为黑白（灰度图）
        if file_suffix in image_extensions:
            img = cv2.imread(str(file_path))
            if img is not None:
                # 转换为灰度图
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 保存图片到目标路径
                cv2.imwrite(str(target_file_path), gray_img)
                print(f"🖼️ 已转换图片: {relative_path}")
            else:
                print(f"⚠️ 无法读取图片，已跳过: {relative_path}")

        # 如果是标签文件，直接复制
        elif file_suffix == '.txt':
            shutil.copy2(file_path, target_file_path)
            print(f"📄 已复制标签: {relative_path}")
            
        # 如果有其他配置文件（比如 classes.txt 或者 dataset.yaml 等），也直接复制
        elif file_suffix in {'.yaml', '.json'}:
            shutil.copy2(file_path, target_file_path)
            print(f"⚙️ 已复制配置文件: {relative_path}")

    print(f"\n✅ 数据集处理完成！已保存至: {output_dir}")


if __name__ == "__main__":
    # ================= 配置区 =================
    # 请在这里填入你的实际输入和输出文件夹路径
    
    INPUT_DATASET_DIR = "/home/ares/depth_visual/src/model_train/database/stage2__merged_bag_15_30_times_1_split"  # 替换为你的原数据集根目录
    OUTPUT_DATASET_DIR = "/home/ares/depth_visual/src/model_train/database/stage2_gray_dataset"                    # 替换为你想要保存的新数据集路径
    # ==========================================

    convert_dataset_to_grayscale(INPUT_DATASET_DIR, OUTPUT_DATASET_DIR)