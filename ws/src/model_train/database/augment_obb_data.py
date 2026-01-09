#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBB Dataset Augmentation Script

This script augments YOLO-OBB datasets by applying various transformations:
- Geometric: rotation, scaling, translation, perspective
- Photometric: HSV adjustment, brightness, contrast
- Flip: horizontal/vertical flip

Input format: YOLO-OBB (class x1 y1 x2 y2 x3 y3 x4 y4, normalized)
Output format: Same as input, with augmented images and labels

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import random
from typing import Tuple, List, Optional
import shutil


class OBBAugmentor:
    """Augment YOLO-OBB dataset with various transformations."""
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        num_augments: int = 5,
        rotation_range: Tuple[float, float] = (-15, 15),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        translate_ratio: float = 0.1,
        perspective_ratio: float = 0.0,
        flip_horizontal: float = 0.5,
        flip_vertical: float = 0.0,
        hsv_h: float = 0.0,
        hsv_s: float = 0.7,
        hsv_v: float = 0.0,
        brightness_range: Tuple[float, float] = (1.0, 1.0),
        contrast_range: Tuple[float, float] = (1.0, 1.0),
        seed: int = 42,
        copy_original: bool = True
    ):
        """
        Initialize the OBB augmentor.
        
        Args:
            source_dir: Path to source YOLO-OBB dataset directory
            output_dir: Path to output directory for augmented dataset
            num_augments: Number of augmented versions per image
            rotation_range: Rotation angle range in degrees (min, max)
            scale_range: Scaling factor range (min, max)
            translate_ratio: Translation ratio relative to image size (0.1 = ±10%)
            perspective_ratio: Perspective distortion ratio (0 = disabled)
            flip_horizontal: Probability of horizontal flip
            flip_vertical: Probability of vertical flip
            hsv_h: HSV hue augmentation (0-1)
            hsv_s: HSV saturation augmentation (0-1)
            hsv_v: HSV value augmentation (0-1)
            brightness_range: Brightness multiplier range
            contrast_range: Contrast multiplier range
            seed: Random seed for reproducibility
            copy_original: Whether to copy original images to output
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.num_augments = num_augments
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translate_ratio = translate_ratio
        self.perspective_ratio = perspective_ratio
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.copy_original = copy_original
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Find source images/labels directories
        split_images_dir = self.source_dir / 'images' / 'train'
        split_labels_dir = self.source_dir / 'labels' / 'train'
        flat_images_dir = self.source_dir / 'images'
        flat_labels_dir = self.source_dir / 'labels'
        
        if split_images_dir.exists() and split_labels_dir.exists():
            self.source_images_dir = split_images_dir
            self.source_labels_dir = split_labels_dir
            self.output_images_dir = self.output_dir / 'images' / 'train'
            self.output_labels_dir = self.output_dir / 'labels' / 'train'
        elif flat_images_dir.exists() and flat_labels_dir.exists():
            self.source_images_dir = flat_images_dir
            self.source_labels_dir = flat_labels_dir
            self.output_images_dir = self.output_dir / 'images'
            self.output_labels_dir = self.output_dir / 'labels'
        else:
            raise ValueError(
                "Unsupported source dataset layout. Expected one of:\n"
                f"  - {split_images_dir} and {split_labels_dir}\n"
                f"  - {flat_images_dir} and {flat_labels_dir}\n"
            )
        
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_obb_label(self, label_path: Path) -> List[Tuple[int, np.ndarray]]:
        """
        Parse YOLO-OBB label file.
        
        Format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1)
        
        Returns:
            List of (class_id, points) tuples, points shape: (4, 2)
        """
        objects = []
        if not label_path.exists():
            return objects
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 9:
                    continue
                
                class_id = int(parts[0])
                coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
                objects.append((class_id, coords))
        
        return objects
    
    def denormalize_points(self, points: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """Convert normalized points (0-1) to image coordinates."""
        pts = points.copy()
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        return pts
    
    def normalize_points(self, points: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """Convert image coordinates to normalized points (0-1)."""
        pts = points.copy()
        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        return np.clip(pts, 0.0, 1.0)
    
    def get_augmentation_matrix(
        self,
        img_shape: Tuple[int, int],
        rotation: float,
        scale: float,
        translate: Tuple[float, float],
        perspective: float
    ) -> np.ndarray:
        """
        Compute augmentation transformation matrix.
        
        Args:
            img_shape: Image shape as (height, width)
            rotation: Rotation angle in degrees
            scale: Scaling factor
            translate: Translation as (tx, ty) in pixels
            perspective: Perspective distortion factor
            
        Returns:
            3x3 transformation matrix
        """
        h, w = img_shape
        center = (w / 2, h / 2)
        
        # Rotation + Scale
        M = cv2.getRotationMatrix2D(center, rotation, scale)
        M = np.vstack([M, [0, 0, 1]])
        
        # Translation
        M[0, 2] += translate[0]
        M[1, 2] += translate[1]
        
        # Perspective (optional)
        if abs(perspective) > 1e-6:
            # Add slight perspective distortion
            src_pts = np.float32([
                [0, 0], [w, 0], [w, h], [0, h]
            ])
            dst_pts = src_pts.copy()
            # Randomly shift corners
            for i in range(4):
                dx = random.uniform(-perspective * w, perspective * w)
                dy = random.uniform(-perspective * h, perspective * h)
                dst_pts[i] += [dx, dy]
            
            M_perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)
            M = M_perspective @ M
        
        return M
    
    def apply_photometric_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply HSV, brightness, and contrast augmentation."""
        img = image.copy()
        
        # HSV augmentation
        if self.hsv_h > 0 or self.hsv_s > 0 or self.hsv_v > 0:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Hue
            img_hsv[:, :, 0] += random.uniform(-self.hsv_h * 180, self.hsv_h * 180)
            img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 180)
            
            # Saturation
            img_hsv[:, :, 1] *= random.uniform(1 - self.hsv_s, 1 + self.hsv_s)
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
            
            # Value
            img_hsv[:, :, 2] *= random.uniform(1 - self.hsv_v, 1 + self.hsv_v)
            img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)
            
            img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Brightness
        brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        
        # Contrast
        contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
        mean = img.mean()
        img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        return img
    
    def transform_image_and_labels(
        self,
        image: np.ndarray,
        objects: List[Tuple[int, np.ndarray]]
    ) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]], bool]:
        """
        Apply random augmentation to image and labels.
        
        Returns:
            augmented_image: Transformed image
            augmented_objects: List of (class_id, transformed_points)
            valid: Whether the augmentation is valid (all objects still visible)
        """
        h, w = image.shape[:2]
        
        # Random flip
        do_flip_h = random.random() < self.flip_horizontal
        do_flip_v = random.random() < self.flip_vertical
        
        img = image.copy()
        new_objects = []
        
        if do_flip_h:
            img = cv2.flip(img, 1)
        if do_flip_v:
            img = cv2.flip(img, 0)
        
        # Flip points
        for class_id, points_norm in objects:
            pts = self.denormalize_points(points_norm, w, h)
            if do_flip_h:
                pts[:, 0] = w - pts[:, 0]
            if do_flip_v:
                pts[:, 1] = h - pts[:, 1]
            new_objects.append((class_id, pts))
        
        # Random geometric transform
        rotation = random.uniform(self.rotation_range[0], self.rotation_range[1])
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        tx = random.uniform(-self.translate_ratio * w, self.translate_ratio * w)
        ty = random.uniform(-self.translate_ratio * h, self.translate_ratio * h)
        
        M = self.get_augmentation_matrix(
            (h, w), rotation, scale, (tx, ty), self.perspective_ratio
        )
        
        # Transform image
        img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        # Transform points
        final_objects = []
        for class_id, pts in new_objects:
            # Add homogeneous coordinate
            pts_homo = np.hstack([pts, np.ones((4, 1))])
            # Transform
            pts_transformed = (M @ pts_homo.T).T
            # Convert from homogeneous
            pts_transformed = pts_transformed[:, :2] / (pts_transformed[:, 2:3] + 1e-8)
            
            # Check if points are within image bounds (with some margin)
            margin = 0.1
            if np.all(pts_transformed >= -margin * np.array([w, h])) and \
               np.all(pts_transformed <= (1 + margin) * np.array([w, h])):
                # Normalize and clip
                pts_norm = self.normalize_points(pts_transformed, w, h)
                final_objects.append((class_id, pts_norm))
        
        # Apply photometric augmentation
        img = self.apply_photometric_augmentation(img)
        
        # Valid if we retained at least one object
        valid = len(final_objects) > 0
        
        return img, final_objects, valid
    
    def process_image(self, image_path: Path, label_path: Path) -> int:
        """
        Process a single image and generate augmented versions.
        
        Returns:
            Number of successfully generated augmented images
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return 0
        
        # Parse labels
        objects = self.parse_obb_label(label_path)
        if len(objects) == 0:
            print(f"Warning: No objects in {label_path}")
            return 0
        
        base_name = image_path.stem
        count = 0
        
        # Copy original if requested
        if self.copy_original:
            output_image = self.output_images_dir / image_path.name
            output_label = self.output_labels_dir / label_path.name
            shutil.copy2(str(image_path), str(output_image))
            shutil.copy2(str(label_path), str(output_label))
            count += 1
        
        # Generate augmentations
        for aug_idx in range(self.num_augments):
            aug_img, aug_objects, valid = self.transform_image_and_labels(image, objects)
            
            if not valid:
                continue
            
            # Save augmented image
            output_image_name = f"{base_name}_aug{aug_idx}{image_path.suffix}"
            output_image_path = self.output_images_dir / output_image_name
            cv2.imwrite(str(output_image_path), aug_img)
            
            # Save augmented label
            output_label_name = f"{base_name}_aug{aug_idx}.txt"
            output_label_path = self.output_labels_dir / output_label_name
            
            with open(output_label_path, 'w') as f:
                for class_id, points in aug_objects:
                    line = f"{class_id}"
                    for pt in points:
                        line += f" {pt[0]:.6f} {pt[1]:.6f}"
                    f.write(line + "\n")
            
            count += 1
        
        return count
    
    def run(self):
        """Run the augmentation process."""
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in self.source_images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if len(image_files) == 0:
            print(f"Error: No images found in {self.source_images_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        print(f"Generating {self.num_augments} augmentations per image")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        total_generated = 0
        
        # Process each image
        for image_path in tqdm(image_files, desc="Augmenting images"):
            label_path = self.source_labels_dir / (image_path.stem + '.txt')
            
            if not label_path.exists():
                continue
            
            count = self.process_image(image_path, label_path)
            total_generated += count
        
        print("-" * 60)
        print(f"Augmentation complete!")
        print(f"  Total images generated: {total_generated}")
        print(f"  Average per source image: {total_generated / len(image_files):.2f}")
        print(f"  Output images: {self.output_images_dir}")
        print(f"  Output labels: {self.output_labels_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Augment YOLO-OBB dataset with various transformations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (5 augmentations per image, auto output to KFS-1_yolo_obb_augmented)
  python augment_obb_data.py --source KFS-1_yolo_obb
  
  # Custom output directory
  python augment_obb_data.py --source KFS-1_yolo_obb --output KFS-1_custom
  
  # More aggressive augmentation
  python augment_obb_data.py \\
      --source KFS-1_yolo_obb \\
      --num-augments 10 \\
      --rotation -30 30 \\
      --scale 0.7 1.3 \\
      --translate 0.15 \\
      --flip-horizontal 0.5 \\
      --flip-vertical 0.1
  
  # Only photometric augmentation (no geometric)
  python augment_obb_data.py \\
      --source KFS-1_yolo_obb \\
      --rotation 0 0 \\
      --scale 1.0 1.0 \\
      --translate 0 \\
      --hsv-h 0.05 \\
      --hsv-s 0.8 \\
      --hsv-v 0.5
        """
    )
    
    parser.add_argument('--source', type=str, required=True,
                       help='Path to source YOLO-OBB dataset directory')
    parser.add_argument('--output', type=str, default=None,
                       help=('Path to output directory for augmented dataset. '
                             'If omitted, uses a sibling directory named <source_dir_name>_augmented.'))
    parser.add_argument('--num-augments', type=int, default=5,
                       help='Number of augmented versions per image (default: 5)')
    parser.add_argument('--rotation', type=float, nargs=2, default=[-15, 15],
                       metavar=('MIN', 'MAX'),
                       help='Rotation angle range in degrees (default: -15 15)')
    parser.add_argument('--scale', type=float, nargs=2, default=[0.8, 1.2],
                       metavar=('MIN', 'MAX'),
                       help='Scaling factor range (default: 0.8 1.2)')
    parser.add_argument('--translate', type=float, default=0.0,
                       help='Translation ratio relative to image size (default: 0.1 = ±10%%)')
    parser.add_argument('--perspective', type=float, default=0.0,
                       help='Perspective distortion ratio (default: 0.0 = disabled)')
    parser.add_argument('--flip-horizontal', type=float, default=0.0,
                       help='Probability of horizontal flip (default: 0.5)')
    parser.add_argument('--flip-vertical', type=float, default=0.0,
                       help='Probability of vertical flip (default: 0.0)')
    parser.add_argument('--hsv-h', type=float, default=0.0,
                       help='HSV hue augmentation (default: 0.0 = disabled)')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                       help='HSV saturation augmentation (default: 0.7)')
    parser.add_argument('--hsv-v', type=float, default=0.0,
                       help='HSV value augmentation (default: 0.0 = disabled)')
    parser.add_argument('--brightness', type=float, nargs=2, default=[1.0, 1.0],
                       metavar=('MIN', 'MAX'),
                       help='Brightness multiplier range (default: 1.0 1.0 = disabled)')
    parser.add_argument('--contrast', type=float, nargs=2, default=[1.0, 1.0],
                       metavar=('MIN', 'MAX'),
                       help='Contrast multiplier range (default: 1.0 1.0 = disabled)')
    parser.add_argument('--no-copy-original', action='store_true',
                       help='Do not copy original images to output (only augmented)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Default output: sibling of source directory, with '_augmented' suffix
    source_dir = Path(args.source)
    if args.output is None:
        output_dir = source_dir.parent / f"{source_dir.name}_augmented"
    else:
        output_dir = Path(args.output)
    
    # Create augmentor
    augmentor = OBBAugmentor(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        num_augments=args.num_augments,
        rotation_range=tuple(args.rotation),
        scale_range=tuple(args.scale),
        translate_ratio=args.translate,
        perspective_ratio=args.perspective,
        flip_horizontal=args.flip_horizontal,
        flip_vertical=args.flip_vertical,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        brightness_range=tuple(args.brightness),
        contrast_range=tuple(args.contrast),
        copy_original=not args.no_copy_original,
        seed=args.seed
    )
    
    # Run
    augmentor.run()


if __name__ == '__main__':
    main()
