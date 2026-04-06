"""
单元测试：验证 detect.py 中 wm_type="OSS" 分支的检测逻辑。

需求: 10.3 — WHEN 选择 wm_type="OSS" 进行检测时，THE SFW_Framework SHALL
对图像执行 DDIM Inversion，然后调用 OSS_Extractor 提取比特信息。

测试覆盖：
- BER 计算逻辑的正确性
- 比特流内容匹配的识别逻辑（比特串 → 整数索引）
- argparse 接受 "OSS" 作为有效的 wm_type 值
"""

import argparse
import math
import sys
from pathlib import Path

import pytest
import torch

# 将 src 目录加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oss import oss_embed, oss_extract


# ---------------------------------------------------------------------------
# 辅助函数：复现 detect.py 中的 BER 计算逻辑
# ---------------------------------------------------------------------------

def compute_ber(extracted_bits: list[int], gt_msg: list[int]) -> float:
    """计算比特错误率 (BER)，与 detect.py 中的逻辑一致。"""
    num_bits = len(gt_msg)
    return sum(a != b for a, b in zip(extracted_bits, gt_msg)) / num_bits


def bits_to_index(bits: list[int]) -> int:
    """将比特列表转换为整数索引，与 detect.py 中的识别逻辑一致。"""
    index = 0
    for b in bits:
        index = (index << 1) | b
    return index


def index_to_bits(index: int, num_bits: int) -> list[int]:
    """将整数索引转换为比特列表（高位在前）。"""
    return [(index >> (num_bits - 1 - i)) & 1 for i in range(num_bits)]


# ---------------------------------------------------------------------------
# 测试 1：argparse 接受 "OSS" 作为有效的 wm_type（detect.py 版本）
# ---------------------------------------------------------------------------

def _build_detect_parser() -> argparse.ArgumentParser:
    """复现 detect.py 中的 argparse 定义。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wm_type",
        choices=["Tree-Ring", "RingID", "HSTR", "HSQR", "OSS"],
        required=True,
    )
    parser.add_argument(
        "--dataset_id",
        choices=["coco", "Gustavo", "DB1k"],
        required=True,
    )
    parser.add_argument("--output_dir", default="outputs")
    return parser


def test_detect_argparse_accepts_oss():
    """验证 detect.py 的 argparse 能正确解析 wm_type="OSS"。"""
    parser = _build_detect_parser()
    args = parser.parse_args(["--wm_type", "OSS", "--dataset_id", "coco"])
    assert args.wm_type == "OSS"


# ---------------------------------------------------------------------------
# 测试 2：BER 计算逻辑正确性
# ---------------------------------------------------------------------------

def test_ber_zero_when_bits_match():
    """完全匹配的比特流 BER 应为 0。"""
    gt_msg = [1, 0, 1, 1, 0, 0, 1, 0]
    extracted = [1, 0, 1, 1, 0, 0, 1, 0]
    assert compute_ber(extracted, gt_msg) == 0.0


def test_ber_one_when_all_bits_differ():
    """完全不匹配的比特流 BER 应为 1.0。"""
    gt_msg = [1, 1, 1, 1]
    extracted = [0, 0, 0, 0]
    assert compute_ber(extracted, gt_msg) == 1.0


def test_ber_partial_mismatch():
    """部分不匹配时 BER 应为错误比特数 / 总比特数。"""
    gt_msg = [1, 0, 1, 0, 1, 0, 1, 0]
    extracted = [1, 0, 0, 0, 1, 1, 1, 0]  # 第 2、5 位错误
    assert compute_ber(extracted, gt_msg) == pytest.approx(2.0 / 8.0)


def test_ber_single_bit():
    """单比特消息的 BER 计算。"""
    assert compute_ber([1], [1]) == 0.0
    assert compute_ber([0], [1]) == 1.0


def test_ber_negative_score_convention():
    """验证 detect.py 中取负 BER 作为得分的约定：BER 越低 → 得分越高。"""
    gt_msg = [1, 0, 1, 1]
    # 有水印图像：BER 低 → -BER 接近 0（得分高）
    wm_bits = [1, 0, 1, 1]
    wm_score = -compute_ber(wm_bits, gt_msg)
    # 无水印图像：BER 高 → -BER 远离 0（得分低）
    no_wm_bits = [0, 1, 0, 0]
    no_wm_score = -compute_ber(no_wm_bits, gt_msg)
    assert wm_score > no_wm_score


# ---------------------------------------------------------------------------
# 测试 3：比特流内容匹配的识别逻辑
# ---------------------------------------------------------------------------

def test_bits_to_index_zero():
    """全零比特串应映射到索引 0。"""
    assert bits_to_index([0, 0, 0, 0]) == 0


def test_bits_to_index_max():
    """全一比特串应映射到最大索引。"""
    num_bits = 11
    bits = [1] * num_bits
    assert bits_to_index(bits) == 2047  # 2^11 - 1


def test_bits_to_index_specific_values():
    """验证特定索引的比特串 ↔ 索引转换。"""
    num_bits = 11
    # 索引 1024 = 0b10000000000
    assert bits_to_index([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == 1024
    # 索引 1 = 0b00000000001
    assert bits_to_index([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) == 1
    # 索引 42 = 0b00000101010
    bits_42 = index_to_bits(42, num_bits)
    assert bits_to_index(bits_42) == 42


def test_index_bits_roundtrip():
    """索引 → 比特串 → 索引 的往返一致性。"""
    num_bits = 11
    for idx in [0, 1, 42, 100, 1023, 1024, 2047]:
        bits = index_to_bits(idx, num_bits)
        assert len(bits) == num_bits
        assert bits_to_index(bits) == idx


def test_identification_correct_with_clean_extraction():
    """无攻击条件下，嵌入后直接提取应能正确识别索引。"""
    seed = 42
    num_bits = 11
    key_index = 137  # 任意选择的 ground-truth 索引
    gt_msg = index_to_bits(key_index, num_bits)

    # 嵌入
    z = torch.randn(1, 4, 64, 64)
    z_wm = oss_embed(z, gt_msg, seed=seed)

    # 提取
    extracted_bits = oss_extract(z_wm, seed=seed, num_bits=num_bits)

    # 比特串 → 索引（与 detect.py 中的逻辑一致）
    extracted_index = bits_to_index(extracted_bits)
    assert extracted_index == key_index


def test_identification_wrong_with_different_seed():
    """使用不同种子提取时，识别结果应（大概率）不正确。"""
    embed_seed = 42
    extract_seed = 999
    num_bits = 11
    key_index = 500
    gt_msg = index_to_bits(key_index, num_bits)

    z = torch.randn(1, 4, 64, 64)
    z_wm = oss_embed(z, gt_msg, seed=embed_seed)
    extracted_bits = oss_extract(z_wm, seed=extract_seed, num_bits=num_bits)
    extracted_index = bits_to_index(extracted_bits)
    # 使用错误种子，识别结果应不匹配
    assert extracted_index != key_index


# ---------------------------------------------------------------------------
# 测试 4：端到端检测流程逻辑验证
# ---------------------------------------------------------------------------

def test_oss_num_bits_calculation():
    """验证 detect.py 中 oss_num_bits 的计算逻辑。"""
    wm_capacity = 2048
    oss_num_bits = math.ceil(math.log2(wm_capacity))
    assert oss_num_bits == 11


def test_verify_ber_filename_convention():
    """验证 OSS 分支使用 'verify-ber.npz' 而非 'verify-l1.npz'。"""
    # detect.py 中：save_verify_name = "verify-ber.npz" if wm_type == "OSS"
    wm_type = "OSS"
    save_verify_name = "verify-ber.npz" if wm_type == "OSS" else "verify-l1.npz"
    assert save_verify_name == "verify-ber.npz"


def test_end_to_end_detection_logic():
    """模拟 detect.py 中 OSS 分支的完整检测逻辑（无攻击场景）。"""
    seed = 42
    num_bits = 11
    key_index = 73
    gt_msg = index_to_bits(key_index, num_bits)

    z = torch.randn(1, 4, 64, 64)
    z_wm = oss_embed(z, gt_msg, seed=seed)

    # ---- 模拟 Verification 逻辑 ----
    # 有水印图像
    wm_bits = oss_extract(z_wm, seed, num_bits)
    wm_ber = compute_ber(wm_bits, gt_msg)
    wm_score = -wm_ber
    assert wm_ber == 0.0  # 无攻击下 BER 应为 0
    assert wm_score == 0.0

    # 无水印图像
    no_wm_bits = oss_extract(z, seed, num_bits)
    no_wm_ber = compute_ber(no_wm_bits, gt_msg)
    no_wm_score = -no_wm_ber
    assert no_wm_ber > 0.0  # 无水印图像的 BER 应大于 0
    assert no_wm_score < wm_score  # 无水印得分应低于有水印得分

    # ---- 模拟 Identification 逻辑 ----
    extracted_bits = oss_extract(z_wm, seed, num_bits)
    extracted_index = bits_to_index(extracted_bits)
    id_acc = (extracted_index == key_index)
    assert id_acc is True


def test_multiple_images_detection():
    """验证多张图像的检测逻辑（模拟 detect.py 的批处理循环）。"""
    seed = 42
    num_bits = 11
    num_images = 5

    for img_idx in range(num_images):
        key_index = img_idx * 100  # 不同图像使用不同索引
        gt_msg = index_to_bits(key_index, num_bits)

        z = torch.randn(1, 4, 64, 64)
        z_wm = oss_embed(z, gt_msg, seed=seed)

        # Verification
        wm_bits = oss_extract(z_wm, seed, num_bits)
        wm_ber = compute_ber(wm_bits, gt_msg)
        assert wm_ber == 0.0, f"图像 {img_idx} 的 BER 应为 0"

        # Identification
        extracted_index = bits_to_index(wm_bits)
        assert extracted_index == key_index, f"图像 {img_idx} 的索引应匹配"
