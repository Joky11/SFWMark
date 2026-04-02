# 需求文档

## 简介

本特性将**正交扩频多比特水印 (Orthogonal Spread Spectrum, OSS)** 集成到现有的 SFW (Symmetric Fourier Watermarking) 框架中。当前 SFW 框架支持基于模式匹配的水印验证与识别（Tree-Ring、RingID、HSTR、HSQR），但不支持在单个潜在变量中嵌入和提取任意长度的多比特信息流。OSS 水印通过在频域 Free half-region 中叠加正交码片（chip）来编码多比特消息，并利用 SFW 已有的 Hermitian 对称填充机制保证 IFFT 后潜在变量为纯实数，从而无缝融入现有的 merged-in-generation 生成流程。

## 术语表

- **SFW_Framework**: 现有的 Symmetric Fourier Watermarking 框架，包含 FFT/IFFT、center-aware 嵌入、Hermitian 对称填充等核心模块（位于 `src/utils.py`）
- **OSS_Embedder**: 正交扩频水印嵌入器，负责将多比特消息编码为频域叠加信号并注入潜在变量
- **OSS_Extractor**: 正交扩频水印提取器，负责从潜在变量的频域表示中通过相关性检测恢复比特信息
- **Key_Generator**: 正交码片生成器，负责生成一组正交或高度不相关的复数码片张量
- **Free_Half_Region**: 在 fftshift 后的频谱中，仅需独立指定值的那一半区域（不包含由 Hermitian 对称性约束的 Restricted half-region）
- **Restricted_Half_Region**: 由 Hermitian 对称性决定的频谱共轭镜像区域，其值由 Free_Half_Region 自动推导
- **Center_Slice**: SFW 框架中的 center-aware 设计，对应 44×44 的中心区域（`start=10, end=54`）
- **Chip**: 单个正交码片，为与嵌入区域形状匹配的复数张量，服从标准复正态分布 CN(0,1)
- **Bipolar_Symbol**: 将二进制比特 {0,1} 映射为 {-1, +1} 的双极性符号
- **Alpha**: 水印嵌入强度参数，控制码片叠加的增益幅度
- **BER**: 比特错误率 (Bit Error Rate)，衡量提取比特与原始比特之间的差异比例
- **Correlation_Detector**: 基于复内积实部的相关性检测器，用于判定每个比特的极性
- **Hermitian_Symmetry_Enforcer**: `enforce_hermitian_symmetry` 函数，确保频域张量满足共轭对称性，使 IFFT 结果为纯实数

## 需求

### 需求 1：代码审计与兼容性确认

**用户故事：** 作为开发者，我希望确认现有 SFW 框架的 FFT、center-aware 嵌入和 Hermitian 对称填充机制能够支持加性叠加操作，以便在不破坏现有功能的前提下集成 OSS 水印。

#### 验收标准

1. THE SFW_Framework SHALL 提供对 Free_Half_Region 索引的访问接口，使 OSS_Embedder 能够确定可独立修改的频域区域
2. WHEN OSS_Embedder 仅修改 Free_Half_Region 中的频域系数时，THE Hermitian_Symmetry_Enforcer SHALL 自动同步更新 Restricted_Half_Region 中的对应共轭值
3. THE SFW_Framework SHALL 保持现有 `inject_wm`、`inject_hsqr`、`fft`、`ifft`、`enforce_hermitian_symmetry` 函数的接口和行为不变
4. THE SFW_Framework SHALL 保持现有 `center_slice`（start=10, end=54）定义不变，使 OSS_Embedder 能够复用相同的 center-aware 偏移逻辑


### 需求 2：正交码片生成

**用户故事：** 作为开发者，我希望能够生成一组可复现的正交复数码片，以便为每个比特位分配独立的扩频序列，实现多比特信息的正交编码。

#### 验收标准

1. THE Key_Generator SHALL 提供函数 `generate_orthogonal_keys(num_bits, shape, seed)`，生成 `num_bits` 个复数码片张量
2. WHEN 给定相同的 `seed` 参数时，THE Key_Generator SHALL 通过 `torch.manual_seed` 生成完全相同的码片序列
3. THE Key_Generator SHALL 生成形状与 SFW_Framework 嵌入区域匹配的码片，每个码片的实部和虚部独立服从标准正态分布 N(0,1)
4. THE Key_Generator SHALL 确保生成的码片之间具有正交性或高度不相关性，使任意两个不同码片的归一化内积绝对值趋近于零
5. WHEN `num_bits` 超过嵌入区域维度所能支持的最大正交码片数量时，THE Key_Generator SHALL 返回描述性错误信息

### 需求 3：多比特水印嵌入

**用户故事：** 作为开发者，我希望将任意长度的二进制消息嵌入到潜在变量的频域中，以便在生成图像中携带多比特水印信息。

#### 验收标准

1. THE OSS_Embedder SHALL 接受潜在变量 `z`（形状为 (N,4,H,W) 的实数张量）和比特流 `msg`（长度为 L 的二进制列表）作为输入
2. WHEN 接收到比特流时，THE OSS_Embedder SHALL 将每个比特 {0,1} 映射为 Bipolar_Symbol {-1, +1}
3. THE OSS_Embedder SHALL 对潜在变量执行 FFT，获取 Center_Slice 对应的频域表示
4. THE OSS_Embedder SHALL 调用 Key_Generator 生成 L 个与 Free_Half_Region 形状匹配的 Chip
5. THE OSS_Embedder SHALL 在 Free_Half_Region 中执行加性叠加：`X_new = X + alpha * sum(b_i * K_i)`，其中 `b_i` 为 Bipolar_Symbol，`K_i` 为第 i 个 Chip，`alpha` 为嵌入强度参数
6. THE OSS_Embedder SHALL 仅修改 Free_Half_Region 中的频域系数，并调用 Hermitian_Symmetry_Enforcer 同步更新 Restricted_Half_Region
7. WHEN 执行 IFFT 后，THE OSS_Embedder SHALL 输出纯实数的水印潜在变量（虚部绝对值最大值小于 1e-5）
8. THE OSS_Embedder SHALL 严格遵守 SFW_Framework 的 `center_slice` 偏移逻辑，确保嵌入区域与现有 center-aware 设计对齐

### 需求 4：Alpha 参数自适应增益

**用户故事：** 作为开发者，我希望嵌入强度参数能够根据潜在变量的统计特性自适应调整，以便在水印鲁棒性和图像质量之间取得平衡。

#### 验收标准

1. THE OSS_Embedder SHALL 支持固定 Alpha 值和自适应 Alpha 值两种模式
2. WHEN 使用自适应模式时，THE OSS_Embedder SHALL 根据潜在变量频域系数的标准差计算 Alpha 值，使水印信号能量与载体信号能量保持可控比例
3. THE OSS_Embedder SHALL 将 Alpha 参数作为可配置参数暴露给调用方，默认值基于经验设定


### 需求 5：多比特水印提取

**用户故事：** 作为开发者，我希望从经过 DDIM Inversion 恢复的潜在变量中提取嵌入的多比特信息，以便验证水印内容。

#### 验收标准

1. THE OSS_Extractor SHALL 接受待检测的潜在变量 `z_hat`（形状为 (N,4,H,W) 的实数张量）和用于生成码片的 `seed` 及 `num_bits` 参数作为输入
2. THE OSS_Extractor SHALL 对潜在变量执行 FFT，获取 Center_Slice 对应的频域表示
3. THE OSS_Extractor SHALL 对频域系数执行均值归一化（减去均值），消除直流偏置对相关性检测的影响
4. THE OSS_Extractor SHALL 使用 Correlation_Detector 计算每个 Chip 与频域系数的复内积实部
5. WHEN 复内积实部大于零时，THE OSS_Extractor SHALL 判定对应比特为 1；WHEN 复内积实部小于等于零时，THE OSS_Extractor SHALL 判定对应比特为 0
6. THE OSS_Extractor SHALL 输出提取的比特列表，长度与嵌入时的 `num_bits` 一致

### 需求 6：调试与诊断支持

**用户故事：** 作为开发者，我希望在嵌入和提取过程中能够查看关键中间数值，以便调试和验证算法的正确性。

#### 验收标准

1. WHEN 启用调试模式时，THE OSS_Embedder SHALL 打印水印叠加前后频域系数的统计信息（均值、标准差）
2. WHEN 启用调试模式时，THE OSS_Extractor SHALL 打印每个比特对应的相关性原始数值
3. WHEN 启用调试模式时，THE OSS_Embedder SHALL 打印 IFFT 后潜在变量虚部的最大绝对值，用于验证 Hermitian 对称性
4. THE OSS_Embedder 和 OSS_Extractor SHALL 通过可选的 `debug` 布尔参数控制调试输出，默认为关闭

### 需求 7：Hermitian 对称性验证

**用户故事：** 作为开发者，我希望验证嵌入水印后的潜在变量经 IFFT 后虚部接近零，以确保 Hermitian 对称性被正确维护。

#### 验收标准

1. WHEN 水印嵌入完成后，THE OSS_Embedder SHALL 验证 IFFT 结果的虚部绝对值最大值小于 1e-5
2. IF IFFT 结果的虚部绝对值最大值大于等于 1e-5，THEN THE OSS_Embedder SHALL 记录警告信息并报告虚部最大绝对值
3. THE OSS_Embedder SHALL 在输出水印潜在变量时取实部，丢弃数值误差产生的微小虚部

### 需求 8：无攻击场景下的比特精确提取

**用户故事：** 作为开发者，我希望在无攻击（无图像后处理）的理想条件下，嵌入的比特信息能够被完全正确地提取，以验证算法的基本正确性。

#### 验收标准

1. WHEN 在无攻击条件下对嵌入了 16 比特消息的潜在变量直接执行提取时，THE OSS_Extractor SHALL 实现 BER 为 0（所有比特完全正确）
2. WHEN 在无攻击条件下对嵌入了 32 比特消息的潜在变量直接执行提取时，THE OSS_Extractor SHALL 实现 BER 为 0（所有比特完全正确）
3. FOR ALL 有效的比特流消息，嵌入后提取的比特流 SHALL 与原始比特流完全一致（往返一致性）

### 需求 9：水印对潜在变量统计分布的影响控制

**用户故事：** 作为开发者，我希望嵌入水印后的潜在变量统计分布不发生剧烈偏移，以保证生成图像的质量不受显著影响。

#### 验收标准

1. WHEN 使用默认 Alpha 参数嵌入水印后，THE OSS_Embedder 输出的潜在变量均值与原始潜在变量均值的差异 SHALL 小于 0.1
2. WHEN 使用默认 Alpha 参数嵌入水印后，THE OSS_Embedder 输出的潜在变量标准差与原始潜在变量标准差的比值 SHALL 在 [0.8, 1.2] 范围内
3. THE OSS_Embedder SHALL 支持通过调整 Alpha 参数来控制水印对潜在变量统计分布的影响程度

### 需求 10：与现有 SFW 生成流程的集成

**用户故事：** 作为开发者，我希望 OSS 水印能够作为新的 `wm_type` 选项无缝集成到现有的生成和检测流程中。

#### 验收标准

1. THE SFW_Framework SHALL 支持 `wm_type="OSS"` 作为新的水印方法选项
2. WHEN 选择 `wm_type="OSS"` 时，THE SFW_Framework SHALL 调用 OSS_Embedder 进行水印嵌入，并将嵌入后的潜在变量传递给 Stable Diffusion 管线生成图像
3. WHEN 选择 `wm_type="OSS"` 进行检测时，THE SFW_Framework SHALL 对图像执行 DDIM Inversion，然后调用 OSS_Extractor 提取比特信息
4. THE OSS_Embedder 和 OSS_Extractor SHALL 遵循与现有水印方法（HSTR、HSQR）相同的 center-aware 设计，使用相同的 `center_slice` 定义
