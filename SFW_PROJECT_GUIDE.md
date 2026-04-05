# SFW 项目上手指南

> 本文件由协助实现 OSS（正交扩频水印）方案的 agent 编写，旨在帮助新 agent 快速熟悉 SFW 框架。

## 项目概述

SFW (Symmetric Fourier Watermarking) 是一个基于 Stable Diffusion 的语义水印框架，发表于 ICCV 2025。核心思路是在扩散模型的潜在空间（latent space）的频域中嵌入水印，然后通过 DDIM Inversion 恢复潜在变量并在频域中检测水印。

论文：https://arxiv.org/abs/2509.07647
仓库：https://github.com/thomas11809/SFWMark

## 环境配置

- Python 3.10，conda 虚拟环境名为 `sfw`
- 激活命令：`conda activate sfw`
- 依赖见 `requirements.txt`，核心依赖：PyTorch 2.5.1、diffusers 0.32.2、transformers
- 安装脚本：`bash install.sh`

## 核心架构

### 文件结构

```
src/
├── utils.py          # 框架核心：FFT/IFFT、Hermitian 对称、水印注入、攻击模拟、评估指标
├── generate.py       # 生成流程：创建水印图像
├── detect.py         # 检测流程：DDIM Inversion + 水印检测
├── metric.py         # 图像质量评估（FID、CLIP Score）
├── oss.py            # [新增] OSS 正交扩频水印模块（我实现的方案）
├── diff_attack/      # 扩散攻击相关代码
├── open_clip/        # CLIP 模型
├── pytorch_fid/      # FID 计算
├── results/          # 结果分析脚本
└── text_dataset/     # 数据集（需下载）
```

### 水印流程（merged-in-generation）

```
生成阶段：
  随机噪声 z ~ N(0,I)  →  频域嵌入水印  →  SD Pipeline 生成图像

检测阶段：
  水印图像  →  DDIM Inversion 恢复 z_hat  →  频域分析  →  水印检测
```

### 现有水印方法

| 方法 | wm_type | 策略 | 特点 |
|------|---------|------|------|
| Tree-Ring | `"Tree-Ring"` | 模式替换 | 环形频域模式 |
| RingID | `"RingID"` | 模式替换 | 多环多通道 |
| HSTR | `"HSTR"` | 模式替换 + center-aware | Hermitian 对称 + 中心区域 |
| HSQR | `"HSQR"` | QR码嵌入 + center-aware | 二维码结构化编码 |

所有方法都遵循 **merged-in-generation** 范式：水印在生成过程中嵌入到初始噪声中。

## 关键模块详解（src/utils.py）

### 全局常量

```python
device = "cuda"
hw_latent = 64                    # 潜在变量空间分辨率
shape = (1, 4, hw_latent, hw_latent)  # 标准潜在变量形状
w_seed = 7433                     # 默认水印种子
RADIUS = 14                       # 环形掩码半径

# Center-aware 设计：44×44 中心区域
start = 10
end = 54
center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))

# 水印容量
wm_capacity = 2048  # 2^(RADIUS - RADIUS_CUTOFF)
```

### FFT/IFFT 接口

```python
def fft(input_tensor):
    # fftshift(fft2(input)) — 将零频移到中心
    return torch.fft.fftshift(torch.fft.fft2(input_tensor), dim=(-1, -2))

def ifft(input_tensor):
    # ifft2(ifftshift(input)) — 逆变换
    return torch.fft.ifft2(torch.fft.ifftshift(input_tensor, dim=(-1, -2)))
```

### Hermitian 对称性（核心机制）

`enforce_hermitian_symmetry(freq_tensor)` 确保频域张量满足共轭对称性 `X[i,j] = conj(X[H-i, W-j])`，使 IFFT 结果为纯实数。

工作原理：
- 以右半/下半象限为"源"（Free Half Region）
- 自动推导左半/上半象限的共轭值（Restricted Half Region）
- DC 点和 Nyquist 点仅保留实部

对于 44×44 的 center_slice 区域：
- Free Half Region = `X[:, :, :, 23:]`，形状 (N, 4, 44, 21)
- 这是你可以自由修改的区域，修改后调用 `enforce_hermitian_symmetry` 即可

### 水印注入函数

```python
# 基于掩码的模式替换（Tree-Ring、RingID、HSTR 使用）
def inject_wm(inverted_latent, w_pattern, w_mask, cut_real=True, center=False, device="cuda")

# HSQR 专用注入
def inject_hsqr(inverted_latent, qr_tensor, center=False, device="cuda", method="original")
```

### DDIM Inversion

```python
def ddim_invert(pipe, image_pil, invert_prompt="", invert_guidance=0):
    # 将图像反转回潜在空间，用于水印检测
```

### 攻击模拟

```python
def image_distortion(img1, img2, seed, **kwargs):
    # 支持的攻击类型：
    # brightness_factor, contrast_factor, jpeg_ratio, gaussian_blur_r,
    # gaussian_std, bm3d_sigma, vaeb_quality, vaec_quality,
    # center_crop_area_ratio, random_crop_area_ratio
```

### 评估指标

```python
def get_distance(tensor1, tensor2, mask, ...)      # L1 距离（验证用）
def get_distance_hsqr(qr_gt_bool, target_fft, ...) # HSQR 专用距离
def get_clip_score(...)                              # CLIP 分数
def get_FID(...)                                     # FID
def get_psnr(...)                                    # PSNR
def get_ssim(...)                                    # SSIM
def get_lpips(...)                                   # LPIPS
```

## generate.py 流程

1. 加载 Stable Diffusion v2.1 pipeline
2. 根据 `wm_type` 预生成 2048 个水印模式（`Fourier_watermark_pattern_list`）
3. 为每个图像分配一个模式索引（`identify_gt_indices`）
4. 批量生成：随机噪声 → 注入水印 → SD Pipeline → 保存图像
5. 同时保存无水印和有水印版本用于对比

## detect.py 流程

1. 加载预生成的水印模式列表
2. 对每张图像施加 12 种攻击（Clean + 11 种攻击）
3. DDIM Inversion 恢复潜在变量
4. 验证（Verification）：计算与 GT 模式的 L1 距离，用于 ROC 曲线
5. 识别（Identification）：遍历所有候选模式，找最近匹配
6. 输出 AUC、MaxAcc、TPR@1%FPR、Id-Acc

## 扩展新水印方法的模式

如果要添加新的 `wm_type`，需要修改：

1. **`src/` 下新建模块**：实现嵌入和提取逻辑
2. **`generate.py`**：在 `wm_type` 分支中添加新选项，处理水印嵌入
3. **`detect.py`**：在检测分支中添加新选项，处理水印提取和评估
4. **`argparse`**：在两个文件的 `--wm_type` choices 中添加新选项

关键约束：
- 嵌入后的潜在变量必须是实数（float32），形状 (N, 4, 64, 64)
- 如果使用 center-aware 设计，操作区域为 `center_slice`（44×44）
- 如果修改频域，必须通过 `enforce_hermitian_symmetry` 确保 IFFT 输出为实数
- 生成的图像需要同时保存无水印和有水印版本

## 测试相关

- 测试框架：pytest + hypothesis（属性测试）
- 测试文件位于 `tests/` 目录
- 运行测试需要设置 `PYTHONPATH=src`：
  ```bash
  conda activate sfw
  PYTHONPATH=src python -m pytest tests/ -v
  ```
- `src/utils.py` 在 import 时会加载 LPIPS 模型（VGG），首次运行会有一些警告，属正常现象

## 实用提示

- `utils.py` 使用 `from utils import *` 的方式导入，所有全局变量和函数都会被导入
- 模型路径在 `generate.py` 中硬编码为 `/data1/lyx/model/stable-diffusion-2-1-base`，可能需要修改
- 数据集需要单独下载，放在 `src/text_dataset/` 下
- `wm_capacity = 2048` 是固定的识别容量
- `center_slice` 是 HSTR/HSQR 的核心设计，将操作限制在中心 44×44 区域以提高裁剪鲁棒性
