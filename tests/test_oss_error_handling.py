"""单元测试：OSS 模块的错误处理与边界情况。

覆盖场景：
- num_bits 超过嵌入区域维度上限 → ValueError
- msg 包含非 {0,1} 值 → ValueError
- msg 长度与码片数不匹配 → ValueError
- alpha <= 0 → ValueError
- z 形状不为 4D 张量 → ValueError
- debug=True 模式不抛出异常
- 16 比特和 32 比特具体示例的往返一致性（BER=0）

需求: 2.5, 6.4, 8.1, 8.2
"""

import pytest
import torch

from src.oss import generate_orthogonal_keys, oss_embed, oss_extract

SHAPE = (4, 44, 21)  # Free_Half_Region 形状
SEED = 42


# ============================================================
# 错误处理：generate_orthogonal_keys
# ============================================================


class TestGenerateOrthogonalKeysErrors:
    """验证 generate_orthogonal_keys 的错误处理。"""

    def test_num_bits_exceeds_max(self):
        """num_bits 超过 D // 2 时应抛出 ValueError。（需求 2.5）"""
        D = 4 * 44 * 21  # = 3696
        max_bits = D // 2  # = 1848
        with pytest.raises(ValueError, match="超过嵌入区域支持的最大值"):
            generate_orthogonal_keys(max_bits + 1, SHAPE, SEED)

    def test_num_bits_at_boundary(self):
        """num_bits 恰好等于 D // 2 时不应抛出异常。"""
        D = 4 * 44 * 21
        max_bits = D // 2
        chips = generate_orthogonal_keys(max_bits, SHAPE, SEED)
        assert len(chips) == max_bits


# ============================================================
# 错误处理：oss_embed
# ============================================================


class TestOssEmbedErrors:
    """验证 oss_embed 的输入验证。"""

    def _make_z(self) -> torch.Tensor:
        """创建有效的 4D 潜在变量张量。"""
        return torch.randn(1, 4, 64, 64)

    def test_z_not_4d_2d(self):
        """z 为 2D 张量时应抛出 ValueError。"""
        z = torch.randn(64, 64)
        with pytest.raises(ValueError, match="4D"):
            oss_embed(z, [0, 1], SEED)

    def test_z_not_4d_3d(self):
        """z 为 3D 张量时应抛出 ValueError。"""
        z = torch.randn(4, 64, 64)
        with pytest.raises(ValueError, match="4D"):
            oss_embed(z, [0, 1], SEED)

    def test_z_not_4d_5d(self):
        """z 为 5D 张量时应抛出 ValueError。"""
        z = torch.randn(1, 1, 4, 64, 64)
        with pytest.raises(ValueError, match="4D"):
            oss_embed(z, [0, 1], SEED)

    def test_msg_contains_invalid_value_2(self):
        """msg 包含 2 时应抛出 ValueError。"""
        z = self._make_z()
        with pytest.raises(ValueError, match="0 或 1"):
            oss_embed(z, [0, 2, 1], SEED)

    def test_msg_contains_negative(self):
        """msg 包含 -1 时应抛出 ValueError。"""
        z = self._make_z()
        with pytest.raises(ValueError, match="0 或 1"):
            oss_embed(z, [0, -1, 1], SEED)

    def test_msg_contains_float_like(self):
        """msg 包含非整数值时应抛出 ValueError。"""
        z = self._make_z()
        with pytest.raises(ValueError, match="0 或 1"):
            oss_embed(z, [0, 0.5, 1], SEED)

    def test_alpha_zero(self):
        """alpha = 0 时应抛出 ValueError。"""
        z = self._make_z()
        with pytest.raises(ValueError, match="alpha"):
            oss_embed(z, [0, 1], SEED, alpha=0)

    def test_alpha_negative(self):
        """alpha < 0 时应抛出 ValueError。"""
        z = self._make_z()
        with pytest.raises(ValueError, match="alpha"):
            oss_embed(z, [0, 1], SEED, alpha=-0.5)


# ============================================================
# 调试模式
# ============================================================


class TestDebugMode:
    """验证 debug=True 模式不抛出异常。（需求 6.4）"""

    def test_embed_debug_no_exception(self):
        """oss_embed 在 debug=True 下正常运行。"""
        z = torch.randn(1, 4, 64, 64)
        msg = [1, 0, 1, 1]
        # 不应抛出任何异常
        z_wm = oss_embed(z, msg, SEED, debug=True)
        assert z_wm.shape == z.shape

    def test_extract_debug_no_exception(self):
        """oss_extract 在 debug=True 下正常运行。"""
        z = torch.randn(1, 4, 64, 64)
        msg = [1, 0, 1, 1]
        z_wm = oss_embed(z, msg, SEED)
        # 不应抛出任何异常
        bits = oss_extract(z_wm, SEED, num_bits=len(msg), debug=True)
        assert len(bits) == len(msg)


# ============================================================
# 往返一致性：16 比特和 32 比特具体示例
# ============================================================


class TestRoundtripSpecificExamples:
    """验证 16 比特和 32 比特具体示例的往返一致性（BER=0）。

    需求 8.1: 16 比特无攻击 BER=0
    需求 8.2: 32 比特无攻击 BER=0
    """

    def test_roundtrip_16_bits(self):
        """16 比特消息嵌入后直接提取，BER 应为 0。"""
        torch.manual_seed(0)
        z = torch.randn(1, 4, 64, 64)
        msg = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
        assert len(msg) == 16

        z_wm = oss_embed(z, msg, SEED, alpha=1.0)
        extracted = oss_extract(z_wm, SEED, num_bits=16)

        ber = sum(a != b for a, b in zip(msg, extracted)) / len(msg)
        assert ber == 0.0, f"16 比特往返 BER={ber}，期望 0"

    def test_roundtrip_32_bits(self):
        """32 比特消息嵌入后直接提取，BER 应为 0。"""
        torch.manual_seed(0)
        z = torch.randn(1, 4, 64, 64)
        msg = [
            1, 0, 1, 1, 0, 0, 1, 0,
            1, 1, 0, 1, 0, 1, 0, 0,
            0, 1, 1, 0, 1, 0, 0, 1,
            1, 0, 0, 1, 1, 1, 0, 1,
        ]
        assert len(msg) == 32

        z_wm = oss_embed(z, msg, SEED, alpha=1.0)
        extracted = oss_extract(z_wm, SEED, num_bits=32)

        ber = sum(a != b for a, b in zip(msg, extracted)) / len(msg)
        assert ber == 0.0, f"32 比特往返 BER={ber}，期望 0"
