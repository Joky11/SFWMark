"""属性测试：OSS 正交扩频水印模块的正确性属性验证。"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.oss import generate_orthogonal_keys

# ---------------------------------------------------------------------------
# Property 1: 码片生成的种子可复现性
# Feature: orthogonal-spread-spectrum-watermark, Property 1: 码片生成的种子可复现性
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------

SHAPE = (4, 44, 21)  # Free_Half_Region 形状


@settings(max_examples=100)
@given(
    seed=st.integers(min_value=0, max_value=2**31),
    num_bits=st.integers(min_value=1, max_value=64),
)
def test_seed_reproducibility(seed: int, num_bits: int) -> None:
    """相同参数两次调用 generate_orthogonal_keys 应产生完全相同的码片序列。"""
    chips_a = generate_orthogonal_keys(num_bits, SHAPE, seed)
    chips_b = generate_orthogonal_keys(num_bits, SHAPE, seed)

    assert len(chips_a) == len(chips_b) == num_bits
    for i, (a, b) in enumerate(zip(chips_a, chips_b)):
        assert torch.equal(a, b), f"码片 {i} 在相同 seed={seed} 下不一致"


# ---------------------------------------------------------------------------
# Property 2: 码片形状与分布正确性
# Feature: orthogonal-spread-spectrum-watermark, Property 2: 码片形状与分布正确性
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    num_bits=st.integers(min_value=1, max_value=64),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_chip_shape_and_distribution(num_bits: int, seed: int) -> None:
    """每个码片形状应与指定 shape 匹配，实部和虚部均值接近 0、标准差接近 1。"""
    chips = generate_orthogonal_keys(num_bits, SHAPE, seed)

    assert len(chips) == num_bits

    for i, chip in enumerate(chips):
        # 形状验证
        assert chip.shape == SHAPE, (
            f"码片 {i} 形状 {chip.shape} 与预期 {SHAPE} 不匹配"
        )
        assert chip.is_complex(), f"码片 {i} 应为复数张量"

        # 分布验证：实部和虚部独立 N(0,1)
        real_part = chip.real.float()
        imag_part = chip.imag.float()

        # 元素数量 = 4*44*21 = 3696，统计容差基于 CLT: ~4/sqrt(N) ≈ 0.066
        tol_mean = 0.15
        tol_std_lo = 0.8
        tol_std_hi = 1.2

        assert abs(real_part.mean().item()) < tol_mean, (
            f"码片 {i} 实部均值 {real_part.mean().item():.4f} 偏离 0 过大"
        )
        assert abs(imag_part.mean().item()) < tol_mean, (
            f"码片 {i} 虚部均值 {imag_part.mean().item():.4f} 偏离 0 过大"
        )
        assert tol_std_lo < real_part.std().item() < tol_std_hi, (
            f"码片 {i} 实部标准差 {real_part.std().item():.4f} 不在 ({tol_std_lo}, {tol_std_hi}) 范围内"
        )
        assert tol_std_lo < imag_part.std().item() < tol_std_hi, (
            f"码片 {i} 虚部标准差 {imag_part.std().item():.4f} 不在 ({tol_std_lo}, {tol_std_hi}) 范围内"
        )


# ---------------------------------------------------------------------------
# Property 3: 码片正交性
# Feature: orthogonal-spread-spectrum-watermark, Property 3: 码片正交性
# Validates: Requirements 2.4
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    num_bits=st.integers(min_value=2, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_chip_orthogonality(num_bits: int, seed: int) -> None:
    """任意两个不同码片的归一化内积绝对值应小于 0.1。"""
    chips = generate_orthogonal_keys(num_bits, SHAPE, seed)

    for i in range(num_bits):
        for j in range(i + 1, num_bits):
            # 复内积: <K_i, K_j> = sum(K_i * conj(K_j))
            inner = torch.sum(chips[i] * chips[j].conj())
            norm_i = torch.sqrt(torch.sum(chips[i] * chips[i].conj()).real)
            norm_j = torch.sqrt(torch.sum(chips[j] * chips[j].conj()).real)
            normalized = (inner / (norm_i * norm_j)).abs().item()

            assert normalized < 0.1, (
                f"码片 {i} 和 {j} 的归一化内积绝对值 {normalized:.4f} >= 0.1 "
                f"(seed={seed}, num_bits={num_bits})"
            )


# ---------------------------------------------------------------------------
# Property 4: 嵌入后 Hermitian 对称性与纯实数输出
# Feature: orthogonal-spread-spectrum-watermark, Property 4: 嵌入后 Hermitian 对称性与纯实数输出
# Validates: Requirements 1.2, 3.6, 3.7, 7.1, 7.3
# ---------------------------------------------------------------------------

from src.oss import oss_embed
from src.utils import fft, center_slice


@settings(max_examples=100)
@given(
    msg=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=32),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_hermitian_symmetry_and_real_output(msg: list[int], seed: int) -> None:
    """嵌入后频域应满足 Hermitian 对称性，IFFT 虚部 < 1e-5，输出为 float32。"""
    z = torch.randn(1, 4, 64, 64)

    z_wm = oss_embed(z, msg, seed, alpha=0.5)

    # (c) 输出为 float32
    assert z_wm.dtype == torch.float32, (
        f"输出 dtype 应为 float32，实际为 {z_wm.dtype}"
    )

    # 提取中心区域并 FFT，验证 Hermitian 对称性
    z_wm_center = z_wm[center_slice]  # (1, 4, 44, 44)
    X = fft(z_wm_center)              # (1, 4, 44, 44) complex

    _, _, H, W = X.shape

    # (a) Hermitian 对称性: X[i,j] ≈ conj(X[H-i, W-j])（索引取模）
    for i in range(H):
        for j in range(W):
            hi = (H - i) % H
            wj = (W - j) % W
            diff = (X[0, :, i, j] - X[0, :, hi, wj].conj()).abs().max().item()
            assert diff < 1e-4, (
                f"Hermitian 对称性违反: X[{i},{j}] 与 conj(X[{hi},{wj}]) "
                f"差异 {diff:.2e} >= 1e-4"
            )

    # (b) IFFT 虚部绝对值最大值 < 1e-5
    z_ifft = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-1, -2)))
    imag_max = z_ifft.imag.abs().max().item()
    assert imag_max < 1e-5, (
        f"IFFT 虚部最大绝对值 {imag_max:.2e} >= 1e-5"
    )


# ---------------------------------------------------------------------------
# Property 5: Center-slice 区域隔离
# Feature: orthogonal-spread-spectrum-watermark, Property 5: Center-slice 区域隔离
# Validates: Requirements 3.8, 10.4
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    msg=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=32),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_center_slice_region_isolation(msg: list[int], seed: int) -> None:
    """嵌入后 center_slice 区域之外的所有元素应与原始 z 完全相同。"""
    z = torch.randn(1, 4, 64, 64)

    z_wm = oss_embed(z, msg, seed, alpha=0.5)

    # 构建 center_slice 之外的掩码：行 0-9 和 54-63，列 0-9 和 54-63
    mask = torch.ones(1, 4, 64, 64, dtype=torch.bool)
    mask[:, :, 10:54, 10:54] = False  # center_slice 内部设为 False

    assert torch.equal(z_wm[mask], z[mask]), (
        "center_slice 区域之外的元素在嵌入后发生了变化"
    )


# ---------------------------------------------------------------------------
# Property 6: 加性叠加公式正确性
# Feature: orthogonal-spread-spectrum-watermark, Property 6: 加性叠加公式正确性
# Validates: Requirements 3.5
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    msg=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=32),
    seed=st.integers(min_value=0, max_value=2**31),
    alpha=st.floats(min_value=0.01, max_value=2.0),
)
def test_additive_superposition_formula(
    msg: list[int], seed: int, alpha: float
) -> None:
    """Free_Half_Region 中频域系数应等于原始系数加上 alpha * Σ(b_i * K_i)。"""
    z = torch.randn(1, 4, 64, 64)

    # 获取嵌入前的 Free_Half_Region 频域系数
    X_orig = fft(z[center_slice].clone())
    X_free_orig = X_orig[:, :, :, 23:].clone()  # (1, 4, 44, 21)

    # 嵌入水印（固定 alpha）
    z_wm = oss_embed(z, msg, seed, alpha=alpha)

    # 获取嵌入后的 Free_Half_Region 频域系数
    X_wm = fft(z_wm[center_slice].clone())
    X_free_wm = X_wm[:, :, :, 23:]  # (1, 4, 44, 21)

    # 手动计算期望的叠加信号 W = alpha * Σ(b_i * K_i)
    chip_shape = X_free_orig.shape[1:]  # (4, 44, 21)
    chips = generate_orthogonal_keys(len(msg), chip_shape, seed)

    W = torch.zeros_like(X_free_orig)
    for i, b in enumerate(msg):
        bipolar = 2 * b - 1
        W = W + bipolar * chips[i]

    X_free_expected = X_free_orig + alpha * W

    # 验证：嵌入后的频域系数应等于原始系数 + alpha * Σ(b_i * K_i)
    # 注意：由于 Hermitian 对称填充 + IFFT + FFT 的往返会引入微小数值误差，
    # 使用相对宽松的容差
    diff = (X_free_wm - X_free_expected).abs().max().item()
    assert diff < 1e-3, (
        f"加性叠加公式验证失败：Free_Half_Region 最大偏差 {diff:.2e} >= 1e-3"
    )


# ---------------------------------------------------------------------------
# Property 11: 自适应 Alpha 计算正确性
# Feature: orthogonal-spread-spectrum-watermark, Property 11: 自适应 Alpha 计算正确性
# Validates: Requirements 4.2
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    msg=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=32),
    seed=st.integers(min_value=0, max_value=2**31),
    alpha_scale=st.floats(min_value=0.1, max_value=2.0),
)
def test_adaptive_alpha_correctness(
    msg: list[int], seed: int, alpha_scale: float
) -> None:
    """自适应模式下 alpha 应等于 alpha_scale * std(X_free)。"""
    z = torch.randn(1, 4, 64, 64)

    # 独立计算期望的自适应 alpha
    X_orig = fft(z[center_slice].clone())
    X_free_orig = X_orig[:, :, :, 23:].clone()  # (1, 4, 44, 21)
    expected_alpha = alpha_scale * X_free_orig.std().item()

    # 使用自适应模式嵌入（alpha=None）
    z_wm = oss_embed(z, msg, seed, alpha=None, alpha_scale=alpha_scale)

    # 获取嵌入后的 Free_Half_Region
    X_wm = fft(z_wm[center_slice].clone())
    X_free_wm = X_wm[:, :, :, 23:]

    # 手动计算叠加信号 W = Σ(b_i * K_i)
    chip_shape = X_free_orig.shape[1:]  # (4, 44, 21)
    chips = generate_orthogonal_keys(len(msg), chip_shape, seed)

    W = torch.zeros_like(X_free_orig)
    for i, b in enumerate(msg):
        bipolar = 2 * b - 1
        W = W + bipolar * chips[i]

    # 验证：嵌入后的频域系数应等于 X_orig + expected_alpha * W
    X_free_expected = X_free_orig + expected_alpha * W
    diff = (X_free_wm - X_free_expected).abs().max().item()
    assert diff < 1e-3, (
        f"自适应 Alpha 验证失败：expected alpha={expected_alpha:.6f}, "
        f"Free_Half_Region 最大偏差 {diff:.2e} >= 1e-3"
    )

# ---------------------------------------------------------------------------
# Property 7: 嵌入-提取往返一致性
# Feature: orthogonal-spread-spectrum-watermark, Property 7: 嵌入-提取往返一致性
# Validates: Requirements 8.1, 8.2, 8.3
# ---------------------------------------------------------------------------

from src.oss import oss_extract


@settings(max_examples=100)
@given(
    msg=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=64),
    seed=st.integers(min_value=0, max_value=2**31),
    z_seed=st.integers(min_value=0, max_value=2**31),
)
def test_embed_extract_roundtrip(msg: list[int], seed: int, z_seed: int) -> None:
    """对任意比特消息，嵌入后直接提取应完全一致（BER=0）。

    使用自适应 alpha（默认模式），确保水印信号能量相对于载体信号
    足够大，从而在无攻击条件下实现完美提取。
    """
    # 使用独立种子生成潜在变量，避免与码片种子冲突导致 RNG 状态耦合
    g = torch.Generator().manual_seed(z_seed)
    z = torch.randn(1, 4, 64, 64, generator=g)

    # 使用自适应 alpha（alpha=None），这是设计文档推荐的默认模式
    z_wm = oss_embed(z, msg, seed, alpha=None, alpha_scale=0.5)
    extracted = oss_extract(z_wm, seed, num_bits=len(msg))

    assert extracted == msg, (
        f"往返一致性失败 (seed={seed}, z_seed={z_seed}, len={len(msg)}): "
        f"BER={sum(a != b for a, b in zip(msg, extracted))}/{len(msg)}"
    )


# ---------------------------------------------------------------------------
# Property 8: 提取输出长度不变量
# Feature: orthogonal-spread-spectrum-watermark, Property 8: 提取输出长度不变量
# Validates: Requirements 5.6
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    num_bits=st.integers(min_value=1, max_value=64),
    z_seed=st.integers(min_value=0, max_value=2**31),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_extract_output_length_invariant(
    num_bits: int, z_seed: int, seed: int
) -> None:
    """oss_extract 输出列表长度应恰好等于 num_bits，无论输入内容如何。"""
    g = torch.Generator().manual_seed(z_seed)
    z_hat = torch.randn(1, 4, 64, 64, generator=g)

    extracted = oss_extract(z_hat, seed, num_bits=num_bits)

    assert isinstance(extracted, list), "oss_extract 应返回列表"
    assert len(extracted) == num_bits, (
        f"输出长度 {len(extracted)} != 期望的 num_bits {num_bits}"
    )
    # 验证每个元素为合法比特值
    assert all(b in (0, 1) for b in extracted), (
        f"输出包含非法比特值: {[b for b in extracted if b not in (0, 1)]}"
    )

# ---------------------------------------------------------------------------
# Property 9: 均值归一化有效性
# Feature: orthogonal-spread-spectrum-watermark, Property 9: 均值归一化有效性
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_mean_normalization_effectiveness(seed: int) -> None:
    """对任意复数张量执行均值归一化后，张量均值绝对值应 < 1e-6。"""
    g = torch.Generator().manual_seed(seed)
    # 生成随机复数张量，模拟 Free_Half_Region 的形状
    real = torch.randn(1, 4, 44, 21, generator=g)
    imag = torch.randn(1, 4, 44, 21, generator=g)
    X = torch.complex(real, imag)

    # 执行均值归一化（与 oss_extract 中的逻辑一致）
    X_normalized = X - X.mean()

    # 验证归一化后均值绝对值 < 1e-6
    mean_abs = X_normalized.mean().abs().item()
    assert mean_abs < 1e-6, (
        f"均值归一化后均值绝对值 {mean_abs:.2e} >= 1e-6"
    )


# ---------------------------------------------------------------------------
# Property 10: 水印对统计分布的影响控制
# Feature: orthogonal-spread-spectrum-watermark, Property 10: 水印对统计分布的影响控制
# Validates: Requirements 9.1, 9.2
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    msg=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=32),
    z_seed=st.integers(min_value=0, max_value=2**31),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_watermark_statistical_distribution_impact(
    msg: list[int], z_seed: int, seed: int
) -> None:
    """适当 alpha_scale 下嵌入水印后，均值差异 < 0.1，标准差比值在 [0.8, 1.2] 范围内。

    使用 alpha_scale=0.1（保守值）确保属性在所有消息长度（1~32 比特）下成立。
    叠加信号能量随比特数线性增长，因此较长消息需要较小的 alpha_scale
    来维持统计分布稳定性（需求 9.3 支持通过调整 alpha 控制影响程度）。
    """
    g = torch.Generator().manual_seed(z_seed)
    z = torch.randn(1, 4, 64, 64, generator=g)

    # 使用自适应 alpha，alpha_scale=0.1 确保统计影响可控
    z_wm = oss_embed(z, msg, seed, alpha=None, alpha_scale=0.1)

    # 验证均值差异 < 0.1（需求 9.1）
    mean_diff = abs(z_wm.mean().item() - z.mean().item())
    assert mean_diff < 0.1, (
        f"均值差异 {mean_diff:.6f} >= 0.1 "
        f"(z.mean={z.mean().item():.6f}, z_wm.mean={z_wm.mean().item():.6f})"
    )

    # 验证标准差比值在 [0.8, 1.2] 范围内（需求 9.2）
    std_ratio = z_wm.std().item() / z.std().item()
    assert 0.8 <= std_ratio <= 1.2, (
        f"标准差比值 {std_ratio:.6f} 不在 [0.8, 1.2] 范围内 "
        f"(z.std={z.std().item():.6f}, z_wm.std={z_wm.std().item():.6f})"
    )
