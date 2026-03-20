import os
import shutil
import random
import argparse
from pathlib import Path
from typing import Optional

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _list_images(images_dir: Path, recursive: bool = False):
    if recursive:
        return [p for p in images_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def split_dataset(
    images_dir: Path,
    labels_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    *,
    recursive: bool = False,
    empty_label_for_missing: bool = True,
):
    """
    将 YOLO 数据集划分为标准的 Train/Val/Test 结构（images/labels 分离）。
    
    Args:
        images_dir: 图片目录（包含图片文件）
        labels_dir: 标签目录（包含与图片同名的 .txt；可为 None，表示输入是“混合目录”）
        output_dir: 输出目录（如果为None，则自动基于源目录名称生成）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir is not None else None

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir 不存在: {images_dir}")
    if labels_dir is not None and not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir 不存在: {labels_dir}")

    # 如果未指定输出目录，基于 images_dir 名称生成
    if output_dir is None:
        # 常见的“分离目录”布局：<root>/images + <root>/labels
        if (
            labels_dir is not None
            and images_dir.name == "images"
            and labels_dir.name == "labels"
            and images_dir.parent == labels_dir.parent
        ):
            root = images_dir.parent
            output_dir = root.parent / f"{root.name}_split"
        else:
            output_dir = images_dir.parent / f"{images_dir.name}_split"
    output_dir = Path(output_dir)
    
    # 1. 准备路径 (Setup Paths)
    # 定义目标拓扑结构
    subsets = ['train', 'val', 'test']
    dirs = ['images', 'labels']
    
    # 创建文件夹结构: output_dir/images/train, output_dir/labels/train 等
    for subset in subsets:
        for d in dirs:
            os.makedirs(output_dir / d / subset, exist_ok=True)

    # 2. 获取全集 (Get the Universal Set)
    images = _list_images(images_dir, recursive=recursive)
    if not images:
        raise ValueError(f"未在 images_dir 找到图片: {images_dir}")
    
    # 3. 随机置换 (Random Permutation)
    # 打乱顺序，保证随机性
    random.shuffle(images)
    
    # 4. 计算切分点 (Calculate Split Indices)
    total_count = len(images)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)
    
    # 划分集合
    splits = {
        'train': images[:train_end],
        'val':   images[train_end:val_end],
        'test':  images[val_end:]
    }

    print(f"Total images: {total_count}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # 5. 执行移动 (Execute Moving)
    copied_labels = 0
    empty_labels = 0
    for subset, files in splits.items():
        for img_path in files:
            # 构建目标路径
            dst_img_path = output_dir / 'images' / subset / img_path.name
            dst_label_path = output_dir / 'labels' / subset / f"{img_path.stem}.txt"

            # 复制图片
            shutil.copy2(img_path, dst_img_path)

            # 查找并复制标签
            if labels_dir is None:
                # “混合目录”模式：标签在 images_dir 同级（同目录）
                src_label_path = images_dir / f"{img_path.stem}.txt"
            else:
                src_label_path = labels_dir / f"{img_path.stem}.txt"

            if src_label_path.exists():
                shutil.copy2(src_label_path, dst_label_path)
                copied_labels += 1
            else:
                if empty_label_for_missing:
                    dst_label_path.write_text("", encoding="utf-8")
                    empty_labels += 1

    print(f"✅ Dataset splitting complete!")
    print(f"输出目录: {output_dir}")
    print(f"Copied labels: {copied_labels}, Empty labels: {empty_labels}")

# --- 使用方法 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="将 YOLO 数据集划分为 Train/Val/Test 结构（支持 images/labels 分离或混合目录）。"
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="数据集根目录（若其下存在 images/ 和 labels/ 则使用分离模式；否则按混合目录处理）",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="图片目录（分离模式）。若提供则优先使用。",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="标签目录（分离模式，可选）。不提供则按混合目录处理。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（不指定则创建 <源目录名>；split 于源目录同级）",
    )
    parser.add_argument("--train", type=float, default=0.8, help="训练集比例，默认 0.8")
    parser.add_argument("--val", type=float, default=0.1, help="验证集比例，默认 0.1")
    parser.add_argument("--test", type=float, default=0.1, help="测试集比例，默认 0.1")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于可复现的随机划分），默认不固定",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描 images_dir 下所有子目录的图片",
    )
    parser.add_argument(
        "--no-empty-label",
        action="store_true",
        help="如果图片缺少标签，不生成空 txt（默认会生成空 txt）",
    )
    args = parser.parse_args()

    # 参数校验
    ratios = (args.train, args.val, args.test)
    if any(r < 0 for r in ratios):
        raise ValueError("train/val/test 的比例必须为非负数")
    s = sum(ratios)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train/val/test 的比例之和必须为 1.0，当前为 {s}")

    if args.seed is not None:
        random.seed(args.seed)

    # 解析输入模式：
    # 1) 显式传 --images-dir (推荐)
    # 2) 传 --source-dir 且其下包含 images/labels
    # 3) 仅传 --source-dir (混合目录：图片和txt同目录)
    if args.images_dir:
        images_dir = Path(args.images_dir)
        labels_dir = Path(args.labels_dir) if args.labels_dir else None
    elif args.source_dir:
        src = Path(args.source_dir)
        if (src / "images").exists():
            images_dir = src / "images"
            labels_dir = (src / "labels") if (src / "labels").exists() else None
        else:
            images_dir = src
            labels_dir = None
    else:
        raise SystemExit("必须提供 --images-dir 或 --source-dir")

    # 默认输出目录：源目录同级 + “；split”
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        name_base = Path(args.source_dir) if args.source_dir else images_dir
        output_dir = name_base.parent / f"{name_base.name}_split"

    split_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        recursive=args.recursive,
        empty_label_for_missing=not args.no_empty_label,
    )