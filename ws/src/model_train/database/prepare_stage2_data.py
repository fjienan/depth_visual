#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Stage 2 Training Data

This script generates a synthetic "cropped" dataset for training the Stage 2 (Pose) model.
It simulates the output of Stage 1 (OBB) by:
1. Computing OBB from 4 GT keypoints
2. Adding jitter/noise to simulate Stage 1 errors
3. Warping images to fixed size
4. Mapping keypoints to warped coordinates

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
import sys


class Stage2DataPreparator:
    """Prepare Stage 2 training data from original dataset."""
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        crop_size: Tuple[int, int] = (256, 256),
        num_variations: int = 10,
        center_jitter: float = 0.05,  # ±5%
        size_scale_range: Tuple[float, float] = (1.1, 1.3),
        angle_jitter: float = 5.0,  # ±5 degrees
        seed: int = 42
    ):
        """
        Initialize the data preparator.
        
        Args:
            source_dir: Path to original YOLO dataset directory
            output_dir: Path to output directory for Stage 2 dataset
            crop_size: Size of cropped images (width, height)
            num_variations: Number of variations per object
            center_jitter: Center position jitter ratio (±5% = 0.05)
            size_scale_range: Range for size scaling (min, max)
            angle_jitter: Angle jitter in degrees (±5°)
            seed: Random seed for reproducibility
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.crop_size = crop_size
        self.num_variations = num_variations
        self.center_jitter = center_jitter
        self.size_scale_range = size_scale_range
        self.angle_jitter = angle_jitter
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Find source images/labels directories.
        # Supported layouts:
        # 1) YOLO split:   source/images/train + source/labels/train
        # 2) Flat layout:  source/images       + source/labels
        split_images_dir = self.source_dir / 'images' / 'train'
        split_labels_dir = self.source_dir / 'labels' / 'train'
        flat_images_dir = self.source_dir / 'images'
        flat_labels_dir = self.source_dir / 'labels'
        
        if split_images_dir.exists() and split_labels_dir.exists():
            self.source_images_dir = split_images_dir
            self.source_labels_dir = split_labels_dir
            # Create output directories (mirror split layout)
            self.output_images_dir = self.output_dir / 'images' / 'train'
            self.output_labels_dir = self.output_dir / 'labels' / 'train'
        elif flat_images_dir.exists() and flat_labels_dir.exists():
            self.source_images_dir = flat_images_dir
            self.source_labels_dir = flat_labels_dir
            # Create output directories (mirror flat layout)
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
    
    def compute_obb_from_keypoints(self, keypoints: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Compute Oriented Bounding Box (OBB) from 4 keypoints using MinAreaRect.
        
        Args:
            keypoints: Array of shape (4, 2) containing 4 (x, y) points in image coordinates
            
        Returns:
            OBB as ((center_x, center_y), (width, height), angle) in OpenCV format
        """
        if keypoints.shape != (4, 2):
            raise ValueError(f"Expected 4 keypoints with shape (4, 2), got {keypoints.shape}")
        
        # Convert to int32 for cv2.minAreaRect
        points_int = keypoints.astype(np.int32)
        
        # Compute minimum area rotated rectangle
        rect = cv2.minAreaRect(points_int)
        
        return rect
    
    def jitter_obb(
        self,
        obb: Tuple[Tuple[float, float], Tuple[float, float], float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Add jitter to OBB to simulate Stage 1 detection errors.
        
        Args:
            obb: Original OBB as ((cx, cy), (w, h), angle)
            
        Returns:
            Jittered OBB
        """
        (cx, cy), (w, h), angle = obb
        
        # Jitter center position (±center_jitter%)
        cx_jitter = cx * (1 + random.uniform(-self.center_jitter, self.center_jitter))
        cy_jitter = cy * (1 + random.uniform(-self.center_jitter, self.center_jitter))
        
        # Jitter size (scale 1.1 ~ 1.3)
        scale = random.uniform(self.size_scale_range[0], self.size_scale_range[1])
        w_jitter = w * scale
        h_jitter = h * scale
        
        # Jitter angle (±angle_jitter degrees)
        angle_jitter = angle + random.uniform(-self.angle_jitter, self.angle_jitter)
        
        return ((cx_jitter, cy_jitter), (w_jitter, h_jitter), angle_jitter)
    
    def obb_to_corners(self, obb: Tuple[Tuple[float, float], Tuple[float, float], float]) -> np.ndarray:
        """
        Convert OBB to 4 corner points.
        
        Args:
            obb: OBB as ((cx, cy), (w, h), angle)
            
        Returns:
            Array of shape (4, 2) containing 4 corner points
        """
        return cv2.boxPoints(obb).astype(np.float32)
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Sort 4 points to: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: Array of shape (4, 2) containing 4 (x, y) points
            
        Returns:
            Ordered array of shape (4, 2)
        """
        if pts.shape != (4, 2):
            raise ValueError(f"Expected 4 points with shape (4, 2), got {pts.shape}")
        
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        
        ordered[0] = pts[np.argmin(s)]      # top-left
        ordered[2] = pts[np.argmax(s)]      # bottom-right
        ordered[1] = pts[np.argmin(diff)]   # top-right
        ordered[3] = pts[np.argmax(diff)]   # bottom-left
        
        return ordered
    
    def warp_image_and_keypoints(
        self,
        image: np.ndarray,
        src_corners: np.ndarray,
        keypoints: np.ndarray,
        dst_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Warp image and transform keypoints using perspective transform.
        
        Args:
            image: Input image (H, W, C)
            src_corners: Source corner points, shape (4, 2) - should be ordered [tl, tr, br, bl]
            keypoints: Keypoints to transform, shape (N, 2) or (N, 3)
            dst_size: Destination size as (width, height)
            
        Returns:
            warped_image: Warped image
            transformed_keypoints: Transformed keypoints in warped coordinates
            M: Perspective transform matrix
            valid: Whether all keypoints are within bounds
        """
        dst_w, dst_h = dst_size
        
        # Define destination points (canonical rectangle)
        dst_corners = np.array([
            [0, 0],                    # top-left
            [dst_w - 1, 0],            # top-right
            [dst_w - 1, dst_h - 1],    # bottom-right
            [0, dst_h - 1]             # bottom-left
        ], dtype=np.float32)
        
        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        # Warp image
        warped_image = cv2.warpPerspective(image, M, dst_size, flags=cv2.INTER_LINEAR)
        
        # Transform keypoints
        if keypoints.shape[1] == 2:
            # Add homogeneous coordinate
            kpts_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
        elif keypoints.shape[1] == 3:
            # Use only x, y for transformation
            kpts_homogeneous = np.hstack([keypoints[:, :2], np.ones((keypoints.shape[0], 1))])
        else:
            raise ValueError(f"Expected keypoints with 2 or 3 columns, got {keypoints.shape[1]}")
        
        # Transform: [N, 3] @ [3, 3]^T = [N, 3]
        transformed_homogeneous = kpts_homogeneous @ M.T
        
        # Convert from homogeneous to Cartesian
        w = transformed_homogeneous[:, 2:3]
        transformed_xy = transformed_homogeneous[:, :2] / (w + 1e-8)
        
        # Check if all points are within bounds
        valid = np.all((transformed_xy >= 0) & (transformed_xy < np.array([dst_w, dst_h])))
        
        # Combine with visibility if original had it
        if keypoints.shape[1] == 3:
            transformed_keypoints = np.hstack([transformed_xy, keypoints[:, 2:3]])
        else:
            transformed_keypoints = transformed_xy
        
        return warped_image, transformed_keypoints, M, valid
    
    def parse_yolo_keypoints(self, label_path: Path) -> List[Tuple[int, np.ndarray]]:
        """
        Parse YOLO label file containing 4-point annotations.
        
        Supported formats (one object per line):
        
        1) 4-point keypoints with visibility (Ultralytics pose-style keypoints):
           class x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
        
        2) 4-point polygon / OBB corners (common in YOLO-OBB datasets):
           class x1 y1 x2 y2 x3 y3 x4 y4
           (visibility will be assumed as 1 for all points)
        
        Args:
            label_path: Path to label file
            
        Returns:
            List of (class_id, keypoints) tuples
            keypoints shape: (4, 3) for (x, y, visibility)
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
                try:
                    class_id = int(parts[0])
                except Exception:
                    continue
                
                # Format (2): class + 8 floats (4 points, no visibility)
                if len(parts) == 9:
                    pts = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
                    v = np.ones((4, 1), dtype=np.float32)
                    kpts_data = np.hstack([pts, v])  # (4, 3)
                    objects.append((class_id, kpts_data))
                    continue
                
                # Format (1): class + 12 floats (4 points with visibility)
                if len(parts) >= 13:
                    kpts_data = np.array([float(x) for x in parts[1:13]], dtype=np.float32).reshape(4, 3)
                    objects.append((class_id, kpts_data))
                    continue
        
        return objects
    
    def denormalize_keypoints(
        self,
        keypoints: np.ndarray,
        img_width: int,
        img_height: int
    ) -> np.ndarray:
        """
        Convert normalized keypoints (0-1) to image coordinates.
        
        Args:
            keypoints: Normalized keypoints, shape (N, 2) or (N, 3)
            img_width: Image width
            img_height: Image height
            
        Returns:
            Keypoints in image coordinates
        """
        kpts_xy = keypoints[:, :2].copy()
        kpts_xy[:, 0] *= img_width
        kpts_xy[:, 1] *= img_height
        
        if keypoints.shape[1] == 3:
            return np.hstack([kpts_xy, keypoints[:, 2:3]])
        return kpts_xy
    
    def normalize_keypoints(
        self,
        keypoints: np.ndarray,
        img_width: int,
        img_height: int
    ) -> np.ndarray:
        """
        Convert image coordinates to normalized keypoints (0-1).
        
        Args:
            keypoints: Keypoints in image coordinates, shape (N, 2) or (N, 3)
            img_width: Image width
            img_height: Image height
            
        Returns:
            Normalized keypoints
        """
        kpts_xy = keypoints[:, :2].copy()
        kpts_xy[:, 0] /= img_width
        kpts_xy[:, 1] /= img_height
        
        # Clamp to [0, 1]
        kpts_xy = np.clip(kpts_xy, 0.0, 1.0)
        
        if keypoints.shape[1] == 3:
            return np.hstack([kpts_xy, keypoints[:, 2:3]])
        return kpts_xy
    
    def process_image(self, image_path: Path, label_path: Path) -> int:
        """
        Process a single image and generate variations.
        
        Args:
            image_path: Path to source image
            label_path: Path to source label file
            
        Returns:
            Number of successfully generated crops
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return 0
        
        img_height, img_width = image.shape[:2]
        
        # Parse labels
        objects = self.parse_yolo_keypoints(label_path)
        if len(objects) == 0:
            return 0
        
        count = 0
        base_name = image_path.stem
        
        # Process each object
        for obj_idx, (class_id, kpts_normalized) in enumerate(objects):
            # Denormalize keypoints to image coordinates
            kpts_image = self.denormalize_keypoints(kpts_normalized, img_width, img_height)
            kpts_xy = kpts_image[:, :2]  # Extract only x, y
            
            # Compute OBB from GT keypoints
            obb = self.compute_obb_from_keypoints(kpts_xy)
            
            # Generate variations
            for var_idx in range(self.num_variations):
                # Jitter OBB
                jittered_obb = self.jitter_obb(obb)
                
                # Convert to corner points
                obb_corners = self.obb_to_corners(jittered_obb)
                
                # Order corners: [tl, tr, br, bl]
                ordered_corners = self.order_points(obb_corners)
                
                # Warp image and transform keypoints
                warped_img, transformed_kpts, M, valid = self.warp_image_and_keypoints(
                    image,
                    ordered_corners,
                    kpts_image,
                    self.crop_size
                )
                
                # Skip if keypoints are out of bounds
                if not valid:
                    continue
                
                # Normalize keypoints relative to crop size
                normalized_kpts = self.normalize_keypoints(
                    transformed_kpts,
                    self.crop_size[0],
                    self.crop_size[1]
                )
                
                # Save image
                output_image_name = f"{base_name}_obj{obj_idx}_var{var_idx}.jpg"
                output_image_path = self.output_images_dir / output_image_name
                cv2.imwrite(str(output_image_path), warped_img)
                
                # Save label
                output_label_name = f"{base_name}_obj{obj_idx}_var{var_idx}.txt"
                output_label_path = self.output_labels_dir / output_label_name
                
                with open(output_label_path, 'w') as f:
                    # Ultralytics Pose label format:
                    #   class cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
                    # where cx,cy,w,h are bbox in YOLO normalized format (relative to crop)
                    xy = normalized_kpts[:, :2]
                    x_min, y_min = xy.min(axis=0)
                    x_max, y_max = xy.max(axis=0)
                    x_min = float(np.clip(x_min, 0.0, 1.0))
                    y_min = float(np.clip(y_min, 0.0, 1.0))
                    x_max = float(np.clip(x_max, 0.0, 1.0))
                    y_max = float(np.clip(y_max, 0.0, 1.0))
                    
                    bw = max(x_max - x_min, 1e-6)
                    bh = max(y_max - y_min, 1e-6)
                    cx = float(np.clip(x_min + bw / 2.0, 0.0, 1.0))
                    cy = float(np.clip(y_min + bh / 2.0, 0.0, 1.0))
                    
                    label_line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                    for kpt in normalized_kpts:
                        x, y = kpt[0], kpt[1]
                        v = int(kpt[2]) if normalized_kpts.shape[1] > 2 else 1
                        label_line += f" {x:.6f} {y:.6f} {v}"
                    f.write(label_line + "\n")
                
                count += 1
        
        return count
    
    def run(self):
        """Run the data preparation process."""
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in self.source_images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if len(image_files) == 0:
            print(f"Error: No images found in {self.source_images_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        print(f"Generating {self.num_variations} variations per object")
        print(f"Output directory: {self.output_dir}")
        print(f"Crop size: {self.crop_size}")
        print("-" * 60)
        
        total_crops = 0
        total_objects = 0
        
        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            label_path = self.source_labels_dir / (image_path.stem + '.txt')
            
            if not label_path.exists():
                continue
            
            # Count objects in this image
            objects = self.parse_yolo_keypoints(label_path)
            total_objects += len(objects)
            
            # Process image
            count = self.process_image(image_path, label_path)
            total_crops += count
        
        print("-" * 60)
        print(f"Processing complete!")
        print(f"  Total objects processed: {total_objects}")
        print(f"  Total crops generated: {total_crops}")
        print(f"  Average crops per object: {total_crops / total_objects if total_objects > 0 else 0:.2f}")
        print(f"  Output images: {self.output_images_dir}")
        print(f"  Output labels: {self.output_labels_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare Stage 2 training data from original dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  # Output defaults to a sibling folder: ../stage2_<source_name>
  python prepare_stage2_data.py --source ../database
  
  # Custom parameters
  python prepare_stage2_data.py \\
      --source ../database \\
      --output ../stage2_database \\
      --crop-size 256 256 \\
      --num-variations 15 \\
      --center-jitter 0.08 \\
      --size-scale 1.15 1.35 \\
      --angle-jitter 8.0
        """
    )
    
    parser.add_argument('--source', type=str, required=True,
                       help='Path to source YOLO dataset directory')
    parser.add_argument('--output', type=str, default=None,
                       help=('Path to output directory for Stage 2 dataset. '
                             'If omitted, uses a sibling directory named stage2_<source_dir_name>.'))
    parser.add_argument('--crop-size', type=int, nargs=2, default=[256, 256],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Size of cropped images (default: 256 256)')
    parser.add_argument('--num-variations', type=int, default=3,
                       help='Number of variations per object (default: 10)')
    parser.add_argument('--center-jitter', type=float, default=0.05,
                       help='Center position jitter ratio (default: 0.05 = ±5%%)')
    parser.add_argument('--size-scale', type=float, nargs=2, default=[1.1, 1.3],
                       metavar=('MIN', 'MAX'),
                       help='Size scaling range (default: 1.1 1.3)')
    parser.add_argument('--angle-jitter', type=float, default=5.0,
                       help='Angle jitter in degrees (default: 5.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()

    # Default output: sibling of source directory, prefixed with 'stage2_'
    source_dir = Path(args.source)
    if args.output is None:
        output_dir = source_dir.parent / f"stage2__{source_dir.name}"
    else:
        output_dir = Path(args.output)
    
    # Create preparator
    preparator = Stage2DataPreparator(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        crop_size=tuple(args.crop_size),
        num_variations=args.num_variations,
        center_jitter=args.center_jitter,
        size_scale_range=tuple(args.size_scale),
        angle_jitter=args.angle_jitter,
        seed=args.seed
    )
    
    # Run
    preparator.run()


if __name__ == '__main__':
    main()
