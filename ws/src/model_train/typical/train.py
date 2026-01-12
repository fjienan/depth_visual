#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Pose 模型训练脚本

这个脚本支持从YAML配置文件读取所有训练参数，也可以通过命令行参数覆盖配置。

使用方法:
    python yolo_train.py --config train_config.yaml
    python yolo_train.py --config train_config.yaml --epochs 100 --batch 32
    python yolo_train.py --model yolov8n-pose.pt --data data.yaml --epochs 500

"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from ultralytics.models.yolo.pose import PoseTrainer
    from ultralytics.utils import LOGGER, colorstr
except ImportError:
    print("错误: 请先安装 ultralytics 库")
    sys.exit(1)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    合并配置文件和命令行参数
    
    Args:
        config: 配置文件字典
        args: 命令行参数
        
    Returns:
        合并后的参数字典，用于传递给PoseTrainer
    """
    # 初始化参数字典
    trainer_args = {}
    
    # ==================== 模型参数 ====================
    if args.model or config.get('model', {}).get('path'):
        trainer_args['model'] = args.model or config.get('model', {}).get('path')
    
    # ==================== 数据参数 ====================
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
    
    # 其他数据参数
    if config.get('data', {}).get('rect'):
        trainer_args['rect'] = config.get('data', {}).get('rect')
    if config.get('data', {}).get('multi_scale'):
        trainer_args['multi_scale'] = config.get('data', {}).get('multi_scale')
    if config.get('data', {}).get('fraction'):
        trainer_args['fraction'] = config.get('data', {}).get('fraction')
    if config.get('data', {}).get('single_cls'):
        trainer_args['single_cls'] = config.get('data', {}).get('single_cls')
    
    # ==================== 训练参数 ====================
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
    if train_config.get('warmup_momentum') is not None:
        trainer_args['warmup_momentum'] = train_config.get('warmup_momentum')
    if train_config.get('warmup_bias_lr') is not None:
        trainer_args['warmup_bias_lr'] = train_config.get('warmup_bias_lr')
    # 注意: lr_scheduler 不是有效的YOLO参数，学习率调度由lrf参数控制
    # if train_config.get('lr_scheduler'):
    #     trainer_args['lr_scheduler'] = train_config.get('lr_scheduler')
    if train_config.get('patience') is not None:
        trainer_args['patience'] = train_config.get('patience')
    if train_config.get('save') is not None:
        trainer_args['save'] = train_config.get('save')
    if train_config.get('save_period') is not None:
        trainer_args['save_period'] = train_config.get('save_period')
    if train_config.get('val') is not None:
        trainer_args['val'] = train_config.get('val')
    if train_config.get('val_period') is not None:
        trainer_args['val_period'] = train_config.get('val_period')
    if train_config.get('plots') is not None:
        trainer_args['plots'] = train_config.get('plots')
    if train_config.get('amp') is not None:
        trainer_args['amp'] = train_config.get('amp')
    if train_config.get('deterministic') is not None:
        trainer_args['deterministic'] = train_config.get('deterministic')
    if train_config.get('sync_bn') is not None:
        trainer_args['sync_bn'] = train_config.get('sync_bn')
    
    # ==================== 损失函数参数 ====================
    loss_config = config.get('loss', {})
    if loss_config.get('box') is not None:
        trainer_args['box'] = loss_config.get('box')
    if loss_config.get('cls') is not None:
        trainer_args['cls'] = loss_config.get('cls')
    if loss_config.get('pose') is not None:
        trainer_args['pose'] = loss_config.get('pose')
    if loss_config.get('dfl') is not None:
        trainer_args['dfl'] = loss_config.get('dfl')
    if loss_config.get('kobj') is not None:
        trainer_args['kobj'] = loss_config.get('kobj')
    if loss_config.get('label_smoothing') is not None:
        trainer_args['label_smoothing'] = loss_config.get('label_smoothing')
    
    # ==================== 输出参数 ====================
    output_config = config.get('output', {})
    if args.project or output_config.get('project'):
        trainer_args['project'] = args.project or output_config.get('project')
    if args.name or output_config.get('name'):
        trainer_args['name'] = args.name or output_config.get('name')
    if output_config.get('exist_ok') is not None:
        trainer_args['exist_ok'] = output_config.get('exist_ok')
    
    # ==================== 设备参数 ====================
    device_config = config.get('device', {})
    if args.device or device_config.get('device'):
        trainer_args['device'] = args.device or device_config.get('device')
    
    # ==================== 其他参数 ====================
    misc_config = config.get('misc', {})
    if misc_config.get('seed') is not None:
        trainer_args['seed'] = misc_config.get('seed')
    if misc_config.get('verbose') is not None:
        trainer_args['verbose'] = misc_config.get('verbose')
    if misc_config.get('resume') is not None:
        trainer_args['resume'] = misc_config.get('resume')
    if misc_config.get('resume_path'):
        trainer_args['resume'] = misc_config.get('resume_path')
    
    # ==================== Pose模型特有参数 ====================
    # 注意: kpt_shape 不是训练参数，应该在数据配置文件(data.yaml)中定义
    # pose_config = config.get('pose', {})
    # if pose_config.get('kpt_shape'):
    #     trainer_args['kpt_shape'] = pose_config.get('kpt_shape')
    
    return trainer_args


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLO Pose 模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用配置文件训练
  python yolo_train.py --config train_config.yaml
  
  # 使用配置文件并覆盖部分参数
  python yolo_train.py --config train_config.yaml --epochs 100 --batch 32
  
  # 直接使用命令行参数（不使用配置文件）
  python yolo_train.py --model yolov8n-pose.pt --data data.yaml --epochs 500 --batch 16
  
  # 从检查点恢复训练
  python yolo_train.py --config train_config.yaml --resume runs/train/yolov8n-pose/weights/last.pt
        """
    )
    
    # 配置文件
    parser.add_argument('--config', type=str, default='train_config.yaml',
                       help='训练配置文件路径 (默认: train_config.yaml)')
    
    # 模型参数
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径或模型名称 (例如: yolov8n-pose.pt)')
    
    # 数据参数
    parser.add_argument('--data', type=str, default=None,
                       help='数据配置文件路径 (YAML格式)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=None,
                       help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--batch', type=int, default=None,
                       help='批次大小 (默认: 16)')
    parser.add_argument('--workers', type=int, default=None,
                       help='数据加载工作进程数 (默认: 8)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数 (默认: 500)')
    parser.add_argument('--patience', type=int, default=None,
                       help='早停耐心值 (默认: 50)')
    parser.add_argument('--lr0', type=float, default=None,
                       help='初始学习率 (默认: 0.01)')
    parser.add_argument('--optimizer', type=str, default=None,
                       choices=['SGD', 'Adam', 'AdamW', 'RMSProp', 'NAdam', 'RAdam'],
                       help='优化器类型 (默认: SGD)')
    
    # 输出参数
    parser.add_argument('--project', type=str, default=None,
                       help='项目输出路径')
    parser.add_argument('--name', type=str, default=None,
                       help='训练运行名称')
    
    # 设备参数
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备 (cpu, 0, 1, 2, ... 或 0,1,2,3)')
    
    # 其他参数
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练 (检查点路径)')
    parser.add_argument('--amp', action='store_true', default=None,
                       help='使用半精度训练 (FP16)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                       help='不使用半精度训练')
    parser.add_argument('--plots', action='store_true', default=None,
                       help='保存训练图表')
    parser.add_argument('--no-plots', dest='plots', action='store_false',
                       help='不保存训练图表')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置文件
    config = {}
    if os.path.exists(args.config):
        try:
            config = load_config(args.config)
            print(f"{colorstr('green', 'bold', '✓')} 成功加载配置文件: {args.config}")
        except Exception as e:
            print(f"{colorstr('red', 'bold', '✗')} 加载配置文件失败: {e}")
            if not args.model or not args.data:
                print("错误: 配置文件加载失败，且未提供必需的 --model 和 --data 参数")
                sys.exit(1)
    else:
        if args.config != 'train_config.yaml':
            print(f"{colorstr('yellow', 'bold', '⚠')} 警告: 配置文件不存在: {args.config}")
        if not args.model or not args.data:
            print("错误: 未提供配置文件，且未提供必需的 --model 和 --data 参数")
            sys.exit(1)
    
    # 合并配置
    trainer_args = merge_configs(config, args)
    
    # 检查必需参数
    if 'model' not in trainer_args:
        print("错误: 未指定模型路径或模型名称")
        print("请在配置文件中设置 model.path 或使用 --model 参数")
        sys.exit(1)
    
    if 'data' not in trainer_args:
        print("错误: 未指定数据配置文件路径")
        print("请在配置文件中设置 data.config 或使用 --data 参数")
        sys.exit(1)
    
    # 打印训练配置摘要
    print("\n" + "="*60)
    print("训练配置摘要")
    print("="*60)
    print(f"模型: {trainer_args.get('model', 'N/A')}")
    print(f"数据: {trainer_args.get('data', 'N/A')}")
    print(f"图像尺寸: {trainer_args.get('imgsz', 'N/A')}")
    print(f"批次大小: {trainer_args.get('batch', 'N/A')}")
    print(f"训练轮数: {trainer_args.get('epochs', 'N/A')}")
    print(f"学习率: {trainer_args.get('lr0', 'N/A')}")
    print(f"优化器: {trainer_args.get('optimizer', 'N/A')}")
    print(f"设备: {trainer_args.get('device', 'N/A')}")
    print(f"输出路径: {trainer_args.get('project', 'N/A')}/{trainer_args.get('name', 'N/A')}")
    print("="*60 + "\n")
    
    # 创建训练器并开始训练
    try:
        trainer = PoseTrainer(overrides=trainer_args)
        trainer.train()
        print(f"\n{colorstr('green', 'bold', '✓')} 训练完成!")
    except KeyboardInterrupt:
        print(f"\n{colorstr('yellow', 'bold', '⚠')} 训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n{colorstr('red', 'bold', '✗')} 训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
