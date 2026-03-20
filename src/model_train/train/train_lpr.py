#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LPR (License Plate Recognition) 两阶段训练脚本

支持训练两阶段级联检测系统：
- Stage 1: YOLOv8-OBB 模型（粗定位）
- Stage 2: YOLOv8-Pose 模型（精细角点检测）

使用方法:
    # 训练 Stage 1 (OBB)
    python train_lpr.py --stage 1 --config stage1_config.yaml
    
    # 训练 Stage 2 (Pose)
    python train_lpr.py --stage 2 --config stage2_config.yaml
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER, colorstr
except ImportError:
    print("错误: 请先安装 ultralytics 库")
    print("安装命令: pip install ultralytics")
    sys.exit(1)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def _slugify(s: str) -> str:
    s = str(s).strip()
    # Keep it filesystem-friendly (ASCII-ish), but avoid heavy dependencies.
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        elif ch in (" ", "/", "\\", ":", "|"):
            out.append("_")
        else:
            # drop other punctuation
            out.append("_")
    # collapse repeats
    res = "".join(out)
    while "__" in res:
        res = res.replace("__", "_")
    return res.strip("_")


def _resolve_path_maybe_relative(path_str: str, base_dir: Path) -> Path:
    p = Path(str(path_str)).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _resolve_project_dir(path_str: str) -> Path:
    """
    Resolve training output project dir relative to this script directory.

    This makes `output.project` stable regardless of current working directory.
    """
    return _resolve_path_maybe_relative(path_str, Path(__file__).resolve().parent)


def _count_images_in_dir(d: Path) -> int:
    if not d.exists() or not d.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    n = 0
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            n += 1
    return n


def _infer_dataset_name_and_size_from_data_yaml(data_yaml_path: Path) -> Tuple[str, Optional[int]]:
    """Infer dataset folder name and total image count from an Ultralytics data yaml."""
    try:
        data_cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return "dataset", None

    # Dataset root
    root = None
    if isinstance(data_cfg, dict) and data_cfg.get("path"):
        root = _resolve_path_maybe_relative(str(data_cfg["path"]), data_yaml_path.parent)
    else:
        root = data_yaml_path.parent

    dataset_name = _slugify(root.name) if root else "dataset"

    if not isinstance(data_cfg, dict):
        return dataset_name, None

    total = 0
    any_split = False
    for key in ("train", "val", "test"):
        v = data_cfg.get(key)
        if not v:
            continue
        any_split = True
        # train/val/test can be a str or list in some Ultralytics configs
        if isinstance(v, (list, tuple)):
            paths = [str(x) for x in v]
        else:
            paths = [str(v)]
        for rel in paths:
            p = _resolve_path_maybe_relative(rel, root)
            total += _count_images_in_dir(p)

    return dataset_name, (total if any_split else None)


def merge_configs(config: Dict[str, Any], args: argparse.Namespace, config_path: str) -> Dict[str, Any]:
    """合并配置文件和命令行参数"""
    trainer_args = {}
    
    # 模型参数
    if args.model or config.get('model', {}).get('path'):
        trainer_args['model'] = args.model or config.get('model', {}).get('path')
    
    # 数据参数
    if args.data or config.get('data', {}).get('config'):
        # Resolve data yaml relative to the stage config file directory.
        stage_cfg_dir = Path(config_path).resolve().parent
        data_cfg_raw = args.data or config.get('data', {}).get('config')
        data_cfg_path = _resolve_path_maybe_relative(str(data_cfg_raw), stage_cfg_dir)
        trainer_args['data'] = str(data_cfg_path)
    
    if args.imgsz or config.get('data', {}).get('imgsz'):
        trainer_args['imgsz'] = args.imgsz or config.get('data', {}).get('imgsz')
    
    if args.batch is not None:
        trainer_args['batch'] = args.batch
    elif config.get('data', {}).get('batch') is not None:
        trainer_args['batch'] = config.get('data', {}).get('batch')
    
    if args.workers is not None:
        trainer_args['workers'] = args.workers
    elif config.get('data', {}).get('workers') is not None:
        trainer_args['workers'] = config.get('data', {}).get('workers')
    
    # 数据增强参数
    aug_config = config.get('data', {}).get('augmentation', {})
    if aug_config:
        trainer_args.update({
            'hsv_h': aug_config.get('hsv_h', 0.015),
            'hsv_s': aug_config.get('hsv_s', 0.7),
            'hsv_v': aug_config.get('hsv_v', 0.4),
            'degrees': aug_config.get('degrees', 0.0),
            'translate': aug_config.get('translate', 0.1),
            'scale': aug_config.get('scale', 0.5),
            'shear': aug_config.get('shear', 0.0),
            'perspective': aug_config.get('perspective', 0.0),
            'flipud': aug_config.get('flipud', 0.0),
            'fliplr': aug_config.get('fliplr', 0.5),
            'mosaic': aug_config.get('mosaic', 1.0),
            'mixup': aug_config.get('mixup', 0.0),
            'copy_paste': aug_config.get('copy_paste', 0.0),
        })
    
    # 训练参数
    if args.epochs is not None:
        trainer_args['epochs'] = args.epochs
    elif config.get('training', {}).get('epochs') is not None:
        trainer_args['epochs'] = config.get('training', {}).get('epochs')
    
    train_config = config.get('training', {})
    if train_config.get('optimizer'):
        trainer_args['optimizer'] = train_config.get('optimizer')
    if train_config.get('lr0') is not None:
        trainer_args['lr0'] = train_config.get('lr0')
    if train_config.get('lrf') is not None:
        trainer_args['lrf'] = train_config.get('lrf')
    if train_config.get('momentum') is not None:
        trainer_args['momentum'] = train_config.get('momentum')
    if train_config.get('weight_decay') is not None:
        trainer_args['weight_decay'] = train_config.get('weight_decay')
    if train_config.get('warmup_epochs') is not None:
        trainer_args['warmup_epochs'] = train_config.get('warmup_epochs')
    if train_config.get('patience') is not None:
        trainer_args['patience'] = train_config.get('patience')
    
    # 输出参数
    output_config = config.get('output', {})
    if args.project or output_config.get('project'):
        project_raw = str(args.project or output_config.get('project'))
        trainer_args['project'] = str(_resolve_project_dir(project_raw))
    if args.name:
        trainer_args['name'] = args.name
    else:
        # Auto naming: dataset folder name + dataset size.
        auto_name = bool(output_config.get("auto_name", False))
        if auto_name and trainer_args.get("data"):
            data_yaml_path = Path(str(trainer_args["data"])).resolve()
            dataset_name, n_imgs = _infer_dataset_name_and_size_from_data_yaml(data_yaml_path)
            stage_name = str(output_config.get("name") or Path(config_path).stem).strip()
            stage_name = _slugify(stage_name) or "run"
            n_str = "unk" if n_imgs is None else str(int(n_imgs))
            template = str(output_config.get("name_template") or "{stage}__{dataset}_n{n}")
            trainer_args["name"] = template.format(stage=stage_name, dataset=dataset_name, n=n_str)
        elif output_config.get('name'):
            trainer_args['name'] = output_config.get('name')
    if output_config.get('exist_ok') is not None:
        trainer_args['exist_ok'] = output_config.get('exist_ok')
    
    # 设备参数
    device_config = config.get('device', {})
    if args.device or device_config.get('device'):
        trainer_args['device'] = args.device or device_config.get('device')
    
    # 其他参数
    misc_config = config.get('misc', {})
    if misc_config.get('seed') is not None:
        trainer_args['seed'] = misc_config.get('seed')
    if misc_config.get('verbose') is not None:
        trainer_args['verbose'] = misc_config.get('verbose')
    if args.resume:
        trainer_args['resume'] = args.resume
    
    return trainer_args


def train_stage1_obb(config_path: str, args: argparse.Namespace):
    """训练 Stage 1: OBB 模型"""
    print("\n" + "="*60)
    print("Stage 1: 训练 OBB 模型（粗定位）")
    print("="*60)
    
    # 加载配置
    config = load_config(config_path)
    trainer_args = merge_configs(config, args, config_path=config_path)
    
    # ---- 兼容性修正（Ultralytics >=8.4.x） ----
    # args.yaml 里可能保存了 multi_scale: 0.0/1.0，Ultralytics 现在要求 bool
    if isinstance(trainer_args.get('multi_scale'), (int, float)):
        trainer_args['multi_scale'] = bool(trainer_args['multi_scale'])

    # 检查必需参数
    if 'model' not in trainer_args:
        print("错误: 未指定模型路径")
        sys.exit(1)
    
    if 'data' not in trainer_args:
        print("错误: 未指定数据配置文件路径")
        sys.exit(1)
    
    # 打印配置摘要
    print(f"模型: {trainer_args.get('model', 'N/A')}")
    print(f"数据: {trainer_args.get('data', 'N/A')}")
    print(f"图像尺寸: {trainer_args.get('imgsz', 'N/A')}")
    print(f"批次大小: {trainer_args.get('batch', 'N/A')}")
    print(f"训练轮数: {trainer_args.get('epochs', 'N/A')}")
    print("="*60 + "\n")
    
    # 创建模型并训练
    try:
        if args.resume:
            resume_ckpt = str(Path(args.resume).expanduser().resolve())
            if not Path(resume_ckpt).exists():
                raise FileNotFoundError(f"resume checkpoint 不存在: {resume_ckpt}")
            print(f"恢复训练: {resume_ckpt}")
            model = YOLO(resume_ckpt)
            trainer_args.pop('model', None)
            # Ultralytics expects boolean resume flag once model is loaded from ckpt.
            trainer_args['resume'] = True
        else:
            model = YOLO(trainer_args['model'])
            trainer_args.pop('model')  # model参数已经在YOLO()中使用了
        
        # 开始训练
        results = model.train(**trainer_args)
        
        print(f"\n{colorstr('green', 'bold', '✓')} Stage 1 训练完成!")
        print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
        return results.save_dir / 'weights' / 'best.pt'
        
    except Exception as e:
        print(f"\n{colorstr('red', 'bold', '✗')} Stage 1 训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def train_stage2_pose(config_path: str, args: argparse.Namespace):
    """训练 Stage 2: Pose 模型"""
    print("\n" + "="*60)
    print("Stage 2: 训练 Pose 模型（精细角点检测）")
    print("="*60)
    
    # 加载配置
    config = load_config(config_path)
    trainer_args = merge_configs(config, args, config_path=config_path)
    
    # ---- 兼容性修正（Ultralytics >=8.4.x） ----
    if isinstance(trainer_args.get('multi_scale'), (int, float)):
        trainer_args['multi_scale'] = bool(trainer_args['multi_scale'])

    # 检查必需参数
    if 'model' not in trainer_args:
        print("错误: 未指定模型路径")
        sys.exit(1)
    
    if 'data' not in trainer_args:
        print("错误: 未指定数据配置文件路径")
        sys.exit(1)
    
    # 打印配置摘要
    print(f"模型: {trainer_args.get('model', 'N/A')}")
    print(f"数据: {trainer_args.get('data', 'N/A')}")
    print(f"图像尺寸: {trainer_args.get('imgsz', 'N/A')}")
    print(f"批次大小: {trainer_args.get('batch', 'N/A')}")
    print(f"训练轮数: {trainer_args.get('epochs', 'N/A')}")
    print("="*60 + "\n")
    
    # 创建模型并训练
    try:
        if args.resume:
            resume_ckpt = str(Path(args.resume).expanduser().resolve())
            if not Path(resume_ckpt).exists():
                raise FileNotFoundError(f"resume checkpoint 不存在: {resume_ckpt}")
            print(f"恢复训练: {resume_ckpt}")
            model = YOLO(resume_ckpt)
            trainer_args.pop('model', None)
            # Ultralytics expects boolean resume flag once model is loaded from ckpt.
            trainer_args['resume'] = True
        else:
            model = YOLO(trainer_args['model'])
            trainer_args.pop('model')
        
        # 开始训练
        results = model.train(**trainer_args)
        
        print(f"\n{colorstr('green', 'bold', '✓')} Stage 2 训练完成!")
        print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
        return results.save_dir / 'weights' / 'best.pt'
        
    except Exception as e:
        print(f"\n{colorstr('red', 'bold', '✗')} Stage 2 训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='LPR 两阶段训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练 Stage 1 (OBB)
  python train_lpr.py --stage 1 --config stage1_config.yaml
  
  # 训练 Stage 2 (Pose)
  python train_lpr.py --stage 2 --config stage2_config.yaml
        """
    )
    
    # 训练阶段选择
    parser.add_argument('--stage', type=int, choices=[1, 2],
                       help='训练阶段: 1=OBB模型, 2=Pose模型')
    
    # 配置文件
    parser.add_argument('--config', type=str,
                       help='训练配置文件路径（单阶段训练时使用）')
    
    # 模型参数
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径或模型名称')
    
    # 数据参数
    parser.add_argument('--data', type=str, default=None,
                       help='数据配置文件路径')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='输入图像尺寸')
    parser.add_argument('--batch', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--workers', type=int, default=None,
                       help='数据加载工作进程数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    
    # 输出参数
    parser.add_argument('--project', type=str, default=None,
                       help='项目输出路径')
    parser.add_argument('--name', type=str, default=None,
                       help='训练运行名称')
    
    # 设备参数
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备')
    
    # 其他参数
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查参数
    if not args.stage:
        print("错误: 必须指定 --stage (1 或 2)")
        sys.exit(1)

    if args.stage == 1:
        # 只训练 Stage 1
        if not args.config:
            print("错误: Stage 1 训练需要 --config 参数")
            sys.exit(1)
        
        train_stage1_obb(args.config, args)
        
    elif args.stage == 2:
        # 只训练 Stage 2
        if not args.config:
            print("错误: Stage 2 训练需要 --config 参数")
            sys.exit(1)
        
        train_stage2_pose(args.config, args)


if __name__ == '__main__':
    main()
