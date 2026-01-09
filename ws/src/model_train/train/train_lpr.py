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
    
    # 完整流程（先训练Stage 1，再准备数据，最后训练Stage 2）
    python train_lpr.py --full-pipeline --stage1-config stage1_config.yaml --stage2-config stage2_config.yaml
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

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


def merge_configs(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """合并配置文件和命令行参数"""
    trainer_args = {}
    
    # 模型参数
    if args.model or config.get('model', {}).get('path'):
        trainer_args['model'] = args.model or config.get('model', {}).get('path')
    
    # 数据参数
    if args.data or config.get('data', {}).get('config'):
        trainer_args['data'] = args.data or config.get('data', {}).get('config')
    
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
        trainer_args['project'] = args.project or output_config.get('project')
    if args.name or output_config.get('name'):
        trainer_args['name'] = args.name or output_config.get('name')
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
    trainer_args = merge_configs(config, args)
    
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
    trainer_args = merge_configs(config, args)
    
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


def prepare_stage2_data(source_dir: str, output_dir: str, config: Dict[str, Any]):
    """准备 Stage 2 训练数据"""
    print("\n" + "="*60)
    print("准备 Stage 2 训练数据")
    print("="*60)
    
    # 导入数据准备模块
    try:
        from prepare_stage2_data import Stage2DataPreparator
    except ImportError:
        print("错误: 无法导入 prepare_stage2_data 模块")
        print("请确保 prepare_stage2_data.py 在同一目录下")
        sys.exit(1)
    
    # 从配置中获取参数
    stage2_data_config = config.get('stage2_data', {})
    crop_size = tuple(stage2_data_config.get('crop_size', [256, 256]))
    num_variations = stage2_data_config.get('num_variations', 10)
    center_jitter = stage2_data_config.get('center_jitter', 0.05)
    size_scale_range = tuple(stage2_data_config.get('size_scale_range', [1.1, 1.3]))
    angle_jitter = stage2_data_config.get('angle_jitter', 5.0)
    
    print(f"源数据目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"裁剪尺寸: {crop_size}")
    print(f"每个对象的变化数: {num_variations}")
    print("="*60 + "\n")
    
    # 创建数据准备器
    preparator = Stage2DataPreparator(
        source_dir=source_dir,
        output_dir=output_dir,
        crop_size=crop_size,
        num_variations=num_variations,
        center_jitter=center_jitter,
        size_scale_range=size_scale_range,
        angle_jitter=angle_jitter
    )
    
    # 执行数据准备
    try:
        preparator.run()
        print(f"\n{colorstr('green', 'bold', '✓')} Stage 2 数据准备完成!")
        return True
    except Exception as e:
        print(f"\n{colorstr('red', 'bold', '✗')} Stage 2 数据准备出错: {e}")
        import traceback
        traceback.print_exc()
        return False


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
  
  # 完整流程（自动执行所有步骤）
  python train_lpr.py --full-pipeline \\
      --stage1-config stage1_config.yaml \\
      --stage2-config stage2_config.yaml \\
      --source-data ./data/original \\
      --stage2-data ./data/stage2
        """
    )
    
    # 训练阶段选择
    parser.add_argument('--stage', type=int, choices=[1, 2],
                       help='训练阶段: 1=OBB模型, 2=Pose模型')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='执行完整流程（Stage 1 -> 数据准备 -> Stage 2）')
    
    # 配置文件
    parser.add_argument('--config', type=str,
                       help='训练配置文件路径（单阶段训练时使用）')
    parser.add_argument('--stage1-config', type=str,
                       help='Stage 1 配置文件路径（完整流程时使用）')
    parser.add_argument('--stage2-config', type=str,
                       help='Stage 2 配置文件路径（完整流程时使用）')
    
    # 数据路径（完整流程时使用）
    parser.add_argument('--source-data', type=str,
                       help='原始数据目录（用于准备Stage 2数据）')
    parser.add_argument('--stage2-data', type=str,
                       help='Stage 2 数据输出目录')
    
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
    if not args.full_pipeline and not args.stage:
        print("错误: 必须指定 --stage 或 --full-pipeline")
        sys.exit(1)
    
    if args.full_pipeline:
        # 完整流程
        if not args.stage1_config or not args.stage2_config:
            print("错误: 完整流程需要 --stage1-config 和 --stage2-config")
            sys.exit(1)
        
        if not args.source_data or not args.stage2_data:
            print("错误: 完整流程需要 --source-data 和 --stage2-data")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("LPR 完整训练流程")
        print("="*60)
        print("步骤 1: 训练 Stage 1 (OBB 模型)")
        print("步骤 2: 准备 Stage 2 数据")
        print("步骤 3: 训练 Stage 2 (Pose 模型)")
        print("="*60)
        
        # 步骤1: 训练 Stage 1
        stage1_model_path = train_stage1_obb(args.stage1_config, args)
        
        # 步骤2: 准备 Stage 2 数据
        stage1_config = load_config(args.stage1_config)
        if not prepare_stage2_data(args.source_data, args.stage2_data, stage1_config):
            print("错误: Stage 2 数据准备失败")
            sys.exit(1)
        
        # 步骤3: 训练 Stage 2
        train_stage2_pose(args.stage2_config, args)
        
        print("\n" + "="*60)
        print(f"{colorstr('green', 'bold', '✓')} LPR 完整训练流程完成!")
        print("="*60)
        print(f"Stage 1 模型: {stage1_model_path}")
        print(f"Stage 2 数据: {args.stage2_data}")
        print("="*60)
        
    elif args.stage == 1:
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
