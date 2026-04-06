# 数据集配置说明

## 已完成的配置

### 1. 创建了 meta_data.json 文件
已从您的MSCOCO数据集中提取了5000个caption，并创建了 `src/text_dataset/coco/meta_data.json` 文件。

**数据来源：**
- COCO annotations文件：`/data1/lyx/dataset/coco/annotations/captions_train2017.json`
- 提取了5000个随机样本（随机种子=42，与论文保持一致）

### 2. 修改了代码以支持自定义路径
修改了 `src/utils.py` 中的 `get_text_dataset` 函数，现在支持通过 `dataset_base_path` 参数指定自定义数据集路径。

## 使用方法

### 生成水印图像
现在您可以直接运行生成脚本：

```bash
cd /home/lu/lyx/SFWMark/src
python generate.py --wm_type HSQR --dataset_id coco
```

### 关于 ground_truth 文件夹

**注意：** 如果您需要计算FID（Fréchet Inception Distance）指标，还需要准备 `ground_truth` 文件夹。

`ground_truth` 文件夹应该包含与 `meta_data.json` 中对应的原始COCO图像。这些图像用于FID计算，用于评估生成图像的质量。

**准备 ground_truth 的方法：**

1. 从 `meta_data.json` 中提取对应的图像文件名
2. 从COCO数据集的 `images` 文件夹中复制这些图像到 `text_dataset/coco/ground_truth/`

如果您暂时不需要计算FID，可以跳过这一步。生成图像和CLIP分数计算不需要ground_truth。

## 重新生成 meta_data.json

如果您需要重新生成 `meta_data.json`（例如使用不同的随机种子或样本数量），可以运行：

```bash
cd /home/lu/lyx/SFWMark/src
python prepare_coco_data.py \
    --coco_annotations /data1/lyx/dataset/coco/annotations/captions_train2017.json \
    --output text_dataset/coco/meta_data.json \
    --num_samples 5000 \
    --seed 42
```

## 文件结构

```
SFWMark/
└── src/
    ├── text_dataset/
    │   └── coco/
    │       ├── meta_data.json          # ✅ 已创建（5000个caption）
    │       └── ground_truth/           # ⚠️ 可选（用于FID计算）
    └── prepare_coco_data.py            # 数据准备脚本
```
