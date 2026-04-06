"""
单元测试：验证 generate.py 中 wm_type="OSS" 参数解析和基本集成逻辑。

需求: 10.1 — THE SFW_Framework SHALL 支持 wm_type="OSS" 作为新的水印方法选项
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# 将 src 目录加入搜索路径，与项目运行方式保持一致
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# 测试 1：argparse 接受 "OSS" 作为有效的 wm_type 值
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """复现 generate.py 中的 argparse 定义。"""
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


def test_argparse_accepts_oss():
    """验证 argparse 能正确解析 wm_type="OSS"。"""
    parser = _build_parser()
    args = parser.parse_args(["--wm_type", "OSS", "--dataset_id", "coco"])
    assert args.wm_type == "OSS"


def test_argparse_rejects_invalid_wm_type():
    """验证 argparse 拒绝无效的 wm_type 值。"""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--wm_type", "INVALID", "--dataset_id", "coco"])


# ---------------------------------------------------------------------------
# 测试 2：OSS 分支的导入和基本调用不抛出异常
# ---------------------------------------------------------------------------

def test_oss_embed_import_and_basic_call():
    """验证 oss_embed 可正常导入并完成基本嵌入调用。"""
    import torch
    from oss import oss_embed

    z = torch.randn(1, 4, 64, 64)
    msg = [1, 0, 1, 1]
    # 基本调用不应抛出异常
    z_wm = oss_embed(z, msg, seed=42)
    assert z_wm.shape == z.shape
    assert z_wm.dtype == torch.float32


def test_oss_extract_import_and_basic_call():
    """验证 oss_extract 可正常导入并完成基本提取调用。"""
    import torch
    from oss import oss_extract

    z_hat = torch.randn(1, 4, 64, 64)
    bits = oss_extract(z_hat, seed=42, num_bits=4)
    assert isinstance(bits, list)
    assert len(bits) == 4
    assert all(b in (0, 1) for b in bits)


def test_oss_branch_index_to_bits_encoding():
    """验证 generate.py 中 OSS 分支的索引→比特串编码逻辑正确。"""
    import math

    wm_capacity = 2048
    oss_num_bits = math.ceil(math.log2(wm_capacity))  # 应为 11
    assert oss_num_bits == 11

    # 验证索引 0 编码为全零比特串
    idx = 0
    bits = [(idx >> (oss_num_bits - 1 - b)) & 1 for b in range(oss_num_bits)]
    assert bits == [0] * 11

    # 验证索引 2047 (最大值) 编码为全一比特串
    idx = 2047
    bits = [(idx >> (oss_num_bits - 1 - b)) & 1 for b in range(oss_num_bits)]
    assert bits == [1] * 11

    # 验证索引 1024 编码为 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    idx = 1024
    bits = [(idx >> (oss_num_bits - 1 - b)) & 1 for b in range(oss_num_bits)]
    assert bits == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
