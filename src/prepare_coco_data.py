#!/usr/bin/env python3
"""
从MSCOCO数据集中提取5000个caption并创建meta_data.json文件
"""
import json
import random
import os
from pathlib import Path

def prepare_coco_meta_data(coco_annotations_path, output_path, num_samples=5000, seed=42):
    """
    从COCO annotations文件中提取指定数量的caption
    
    Args:
        coco_annotations_path: COCO annotations JSON文件路径
        output_path: 输出meta_data.json的路径
        num_samples: 要提取的样本数量（默认5000）
        seed: 随机种子（默认42）
    """
    print(f"正在读取COCO annotations文件: {coco_annotations_path}")
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    annotations = coco_data.get('annotations', [])
    images = coco_data.get('images', [])
    
    print(f"找到 {len(annotations)} 个annotations")
    
    # 设置随机种子并随机选择5000个样本
    random.seed(seed)
    selected_annotations = random.sample(annotations, min(num_samples, len(annotations)))
    
    # 获取对应的image_id集合
    selected_image_ids = set(ann['image_id'] for ann in selected_annotations)
    
    # 提取对应的images
    selected_images = [img for img in images if img['id'] in selected_image_ids]
    
    # 创建输出数据结构
    output_data = {
        'images': selected_images,
        'annotations': selected_annotations
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存到文件
    print(f"正在保存到: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"成功创建meta_data.json，包含 {len(selected_annotations)} 个annotations")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从COCO数据集中提取caption")
    parser.add_argument("--coco_annotations", 
                       default="/data1/lyx/dataset/coco/annotations/captions_train2017.json",
                       help="COCO annotations JSON文件路径")
    parser.add_argument("--output", 
                       default="text_dataset/coco/meta_data.json",
                       help="输出meta_data.json路径（相对于src目录）")
    parser.add_argument("--num_samples", type=int, default=5000,
                       help="要提取的样本数量")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 获取脚本所在目录（src目录）
    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    
    prepare_coco_meta_data(
        coco_annotations_path=args.coco_annotations,
        output_path=str(output_path),
        num_samples=args.num_samples,
        seed=args.seed
    )
