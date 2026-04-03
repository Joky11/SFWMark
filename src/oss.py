"""
正交扩频 (Orthogonal Spread Spectrum, OSS) 多比特水印模块。

通过在频域 Free Half-Region 中加性叠加正交码片来编码多比特消息，
复用现有 SFW 框架的 FFT/IFFT 和 Hermitian 对称填充机制。
"""

from __future__ import annotations

import math
import warnings

import torch

from utils import fft, ifft, enforce_hermitian_symmetry, center_slice


def generate_orthogonal_keys(
    num_bits: int,
    shape: tuple,
    seed: int,
) -> list[torch.Tensor]:
    """生成 num_bits 个复正态正交码片张量。

    每个码片从 CN(0, 1) 中采样——实部和虚部独立服从标准正态分布——
    使用确定性种子，确保相同的 (num_bits, shape, seed) 三元组
    始终产生完全相同的码片序列。

    参数:
        num_bits: 码片数量（每个比特对应一个码片）。
        shape:    每个码片的形状，与 Free_Half_Region 匹配
                  （如 ``(4, 44, 21)``）。
        seed:     随机种子，传递给 ``torch.manual_seed``。

    返回:
        长度为 num_bits 的列表，每个元素为指定 shape 的 complex64 张量。

    异常:
        ValueError: 当 num_bits 超过 ``prod(shape) // 2`` 时抛出。
    """
    D = math.prod(shape)
    if num_bits > D // 2:
        raise ValueError(
            f"num_bits ({num_bits}) 超过嵌入区域支持的最大值 "
            f"(D // 2 = {D // 2}，其中 D = prod{shape} = {D})"
        )

    torch.manual_seed(seed)
    chips: list[torch.Tensor] = []
    for _ in range(num_bits):
        real = torch.randn(shape)
        imag = torch.randn(shape)
        chips.append(torch.complex(real, imag))
    return chips


def oss_embed(
    z: torch.Tensor,
    msg: list[int],
    seed: int,
    alpha: float | None = None,
    alpha_scale: float = 0.5,
    debug: bool = False,
) -> torch.Tensor:
    """将多比特消息嵌入潜在变量的频域 Free_Half_Region。

    参数:
        z:           (N, 4, H, W) 实数潜在变量。
        msg:         长度为 L 的二进制列表，元素仅含 {0, 1}。
        seed:        码片生成种子。
        alpha:       嵌入强度。``None`` 表示自适应模式。
        alpha_scale:  自适应模式下的缩放因子，默认 0.5。
        debug:       是否打印调试信息。

    返回:
        z_wm — 与 z 同形状的实数水印潜在变量 (float32)。
    """
    # ---- 输入验证 ----
    if z.dim() != 4:
        raise ValueError(f"z 必须为 4D 张量，当前维度为 {z.dim()}D")
    if not all(b in (0, 1) for b in msg):
        raise ValueError("msg 中的元素必须为 0 或 1")
    if alpha is not None and alpha <= 0:
        raise ValueError(f"固定模式下 alpha 必须 > 0，当前值为 {alpha}")

    L = len(msg)

    # ---- 提取中心区域并 FFT ----
    z_center = z[center_slice].clone()          # (N, 4, 44, 44)
    X = fft(z_center)                           # (N, 4, 44, 44) complex

    # ---- Free_Half_Region ----
    X_free = X[:, :, :, 23:].clone()            # (N, 4, 44, 21)

    # ---- 生成码片 ----
    chip_shape = X_free.shape[1:]               # (4, 44, 21)
    chips = generate_orthogonal_keys(L, chip_shape, seed)

    # ---- 调试：叠加前统计 ----
    if debug:
        print(f"[OSS embed] X_free 叠加前 — mean: {X_free.mean():.6f}, std: {X_free.std():.6f}")

    # ---- 计算 alpha ----
    if alpha is None:
        std_val = X_free.std().item()
        alpha = alpha_scale * std_val if std_val > 0 else 0.5
    if debug:
        print(f"[OSS embed] alpha = {alpha:.6f}")

    # ---- 比特映射 {0,1} → {-1,+1} 并叠加 ----
    W = torch.zeros_like(X_free)
    for i, b in enumerate(msg):
        bipolar = 2 * b - 1
        W = W + bipolar * chips[i].to(W.device)
    X_free_new = X_free + alpha * W

    # ---- 调试：叠加后统计 ----
    if debug:
        print(f"[OSS embed] X_free 叠加后 — mean: {X_free_new.mean():.6f}, std: {X_free_new.std():.6f}")

    # ---- 回写 Free_Half_Region ----
    X[:, :, :, 23:] = X_free_new

    # ---- Hermitian 对称填充 ----
    X_sym = enforce_hermitian_symmetry(X)

    # ---- IFFT 并验证虚部 ----
    z_wm_center_complex = ifft(X_sym)
    imag_max = z_wm_center_complex.imag.abs().max().item()
    if debug:
        print(f"[OSS embed] IFFT 虚部最大绝对值: {imag_max:.2e}")
    if imag_max >= 1e-5:
        warnings.warn(
            f"IFFT 后虚部最大绝对值 ({imag_max:.2e}) >= 1e-5，"
            "Hermitian 对称性可能未被正确维护"
        )

    z_wm_center = z_wm_center_complex.real.float()

    # ---- 回写中心区域 ----
    z_wm = z.clone()
    z_wm[center_slice] = z_wm_center
    return z_wm
