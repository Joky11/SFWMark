# 实现计划：正交扩频多比特水印 (OSS)

## 概述

将正交扩频多比特水印 (OSS) 集成到现有 SFW 框架中。实现分为四个阶段：核心模块开发（`src/oss.py`）、集成到生成/检测流程、属性测试验证、最终集成检查。所有代码使用 Python，基于 PyTorch 和 hypothesis 测试库。

## 任务

- [ ] 1. 实现 Key_Generator 正交码片生成器
  - [x] 1.1 在 `src/oss.py` 中创建 `generate_orthogonal_keys(num_bits, shape, seed)` 函数
    - 使用 `torch.manual_seed(seed)` 控制随机数生成器
    - 从标准复正态分布 CN(0,1) 中采样 `num_bits` 个码片张量（实部和虚部独立 N(0,1)）
    - 每个码片形状为 `shape`（如 `(4, 44, 21)` 对应 Free_Half_Region）
    - 当 `num_bits > D // 2`（D = prod(shape)）时抛出 `ValueError`，包含描述性错误信息
    - _需求: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 1.2 编写属性测试：码片生成的种子可复现性
    - **Property 1: 码片生成的种子可复现性**
    - 使用相同参数两次调用 `generate_orthogonal_keys`，验证产生完全相同的码片序列
    - 生成器：`st.integers(0, 2**31)` 用于 seed，`st.integers(1, 64)` 用于 num_bits
    - **验证: 需求 2.2**

  - [x] 1.3 编写属性测试：码片形状与分布正确性
    - **Property 2: 码片形状与分布正确性**
    - 验证每个码片形状与指定 `shape` 匹配，实部和虚部均值接近 0、标准差接近 1
    - 生成器：`st.integers(1, 64)` 用于 num_bits，`st.integers(0, 2**31)` 用于 seed
    - **验证: 需求 2.3**

  - [x] 1.4 编写属性测试：码片正交性
    - **Property 3: 码片正交性**
    - 验证任意两个不同码片的归一化内积绝对值小于 0.1
    - 生成器：`st.integers(2, 32)` 用于 num_bits，`st.integers(0, 2**31)` 用于 seed
    - **验证: 需求 2.4**


- [x] 2. 实现 OSS_Embedder 水印嵌入器
  - [x] 2.1 在 `src/oss.py` 中实现 `oss_embed(z, msg, seed, alpha, alpha_scale, debug)` 函数
    - 输入验证：检查 `z` 为 4D 张量、`msg` 仅含 {0,1}、`alpha > 0`（固定模式时）
    - 提取中心区域 `z[center_slice]`（44×44），执行 FFT 获取频域表示
    - 提取 Free_Half_Region `X[:, :, :, 23:]`（形状 (N,4,44,21)）
    - 调用 `generate_orthogonal_keys` 生成 L 个码片
    - 比特映射 {0,1} → {-1,+1}，计算叠加信号 `W = alpha * Σ(b_i * K_i)`
    - 支持固定 alpha 和自适应 alpha（`alpha_scale * std(X_free)`）两种模式
    - 回写 Free_Half_Region，调用 `enforce_hermitian_symmetry` 同步 Restricted_Half_Region
    - IFFT 后验证虚部最大绝对值 < 1e-5，若超出则 `warnings.warn`
    - 取实部输出，回写到 `z_wm[center_slice]`
    - 从 `src/utils.py` 导入 `fft`, `ifft`, `enforce_hermitian_symmetry`, `center_slice`
    - _需求: 1.1, 1.2, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4.1, 4.2, 4.3, 7.1, 7.2, 7.3_

  - [x] 2.2 实现调试模式输出
    - 当 `debug=True` 时打印叠加前后 `X_free` 的均值和标准差
    - 打印使用的 alpha 值
    - 打印 IFFT 后虚部最大绝对值
    - _需求: 6.1, 6.3, 6.4_

  - [x] 2.3 编写属性测试：嵌入后 Hermitian 对称性与纯实数输出
    - **Property 4: 嵌入后 Hermitian 对称性与纯实数输出**
    - 验证嵌入后频域满足 Hermitian 对称性，IFFT 虚部 < 1e-5，输出为 float32
    - 生成器：随机 float 张量 + `st.lists(st.integers(0,1), min_size=1, max_size=32)`
    - **验证: 需求 1.2, 3.6, 3.7, 7.1, 7.3**

  - [x] 2.4 编写属性测试：Center-slice 区域隔离
    - **Property 5: Center-slice 区域隔离**
    - 验证嵌入后 `center_slice` 区域之外的所有元素与原始 `z` 完全相同
    - 生成器：随机 float 张量
    - **验证: 需求 3.8, 10.4**

  - [x] 2.5 编写属性测试：加性叠加公式正确性
    - **Property 6: 加性叠加公式正确性**
    - 验证 Free_Half_Region 中频域系数等于原始系数加上 `alpha * Σ(b_i * K_i)`
    - 生成器：随机张量 + 随机消息 + 固定 alpha
    - **验证: 需求 3.5**

  - [x] 2.6 编写属性测试：自适应 Alpha 计算正确性
    - **Property 11: 自适应 Alpha 计算正确性**
    - 验证自适应模式下 `alpha = alpha_scale * std(X_free)`
    - 生成器：随机 float 张量 + `st.floats(0.1, 2.0)`
    - **验证: 需求 4.2**

- [x] 3. 检查点 - 确保嵌入器核心功能正确
  - 确保所有测试通过，如有疑问请向用户确认。


- [-] 4. 实现 OSS_Extractor 水印提取器
  - [x] 4.1 在 `src/oss.py` 中实现 `oss_extract(z_hat, seed, num_bits, debug)` 函数
    - 提取中心区域并 FFT，获取 Free_Half_Region `X_hat[:, :, :, 23:]`
    - 均值归一化：`X_hat_free -= X_hat_free.mean()`，消除直流偏置
    - 重新生成码片（使用相同 seed 和 num_bits）
    - 对每个码片计算相关性：`corr_i = Re(Σ X_hat_free * conj(K_i))`
    - 比特判定：`corr_i > 0` → 1，否则 → 0
    - 输出长度为 `num_bits` 的比特列表
    - 从 `src/utils.py` 导入 `fft`, `center_slice`
    - _需求: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x] 4.2 实现提取器调试模式
    - 当 `debug=True` 时打印每个比特对应的相关性原始数值
    - 打印均值归一化前后的频域统计信息
    - _需求: 6.2, 6.4_

  - [x] 4.3 编写属性测试：嵌入-提取往返一致性
    - **Property 7: 嵌入-提取往返一致性**
    - 对任意比特消息（长度 1~64），嵌入后直接提取应完全一致（BER=0）
    - 生成器：`st.lists(st.integers(0,1), min_size=1, max_size=64)` + 随机 float 张量
    - **验证: 需求 8.1, 8.2, 8.3**

  - [x] 4.4 编写属性测试：提取输出长度不变量
    - **Property 8: 提取输出长度不变量**
    - 验证 `oss_extract` 输出列表长度恰好等于 `num_bits`
    - 生成器：`st.integers(1, 64)` + 随机 float 张量
    - **验证: 需求 5.6**

  - [x] 4.5 编写属性测试：均值归一化有效性
    - **Property 9: 均值归一化有效性**
    - 验证归一化后张量均值绝对值 < 1e-6
    - 生成器：随机 complex 张量
    - **验证: 需求 5.3**

  - [x] 4.6 编写属性测试：水印对统计分布的影响控制
    - **Property 10: 水印对统计分布的影响控制**
    - 验证默认 alpha 下均值差异 < 0.1，标准差比值在 [0.8, 1.2] 范围内
    - 生成器：随机 float 张量 + 随机消息
    - **验证: 需求 9.1, 9.2**

- [x] 5. 检查点 - 确保提取器和往返一致性正确
  - 确保所有测试通过，如有疑问请向用户确认。


- [ ] 6. 集成到 generate.py 生成流程
  - [x] 6.1 在 `src/generate.py` 中添加 `wm_type="OSS"` 分支
    - 在 argparse 的 `--wm_type` choices 中添加 `"OSS"` 选项
    - 在水印模式生成分支中添加 OSS 逻辑：`from oss import oss_embed`
    - OSS 不需要预生成 `Fourier_watermark_pattern_list`，改为为每个图像生成唯一的比特消息
    - 使用 `identify_gt_indices` 映射为每个图像分配唯一的比特流（将索引转换为二进制比特串）
    - 在批处理循环中调用 `oss_embed(no_watermark_latents, msg, seed)` 生成水印潜在变量
    - 保存每个图像对应的比特消息用于后续检测验证
    - _需求: 10.1, 10.2, 10.4_

  - [x] 6.2 编写单元测试：验证 `wm_type="OSS"` 参数解析和基本集成逻辑
    - 测试 argparse 接受 `"OSS"` 作为有效的 `wm_type` 值
    - 测试 OSS 分支的导入和基本调用不抛出异常
    - _需求: 10.1_

- [x] 7. 集成到 detect.py 检测流程
  - [x] 7.1 在 `src/detect.py` 中添加 `wm_type="OSS"` 分支
    - 在检测评估方法分支中添加 OSS 逻辑：`from oss import oss_extract`
    - DDIM Inversion 后调用 `oss_extract` 提取比特信息
    - 计算 BER（比特错误率）替代 L1 距离作为验证指标
    - 识别通过比特流内容匹配实现，无需遍历所有候选模式
    - 适配现有的攻击测试框架，对每种攻击场景计算 BER
    - 输出结果格式与现有方法保持一致（表格打印）
    - _需求: 10.3, 10.4_

  - [ ]* 7.2 编写单元测试：验证检测流程的 OSS 分支逻辑
    - 测试 BER 计算逻辑的正确性
    - 测试比特流内容匹配的识别逻辑
    - _需求: 10.3_

- [ ] 8. 编写单元测试：错误处理与边界情况
  - [ ] 8.1 编写单元测试覆盖错误处理场景
    - 测试 `num_bits` 超过嵌入区域维度上限时抛出 `ValueError`
    - 测试 `msg` 包含非 {0,1} 值时抛出 `ValueError`
    - 测试 `msg` 长度与码片数不匹配时抛出 `ValueError`
    - 测试 `alpha <= 0` 时抛出 `ValueError`
    - 测试 `z` 形状不为 4D 张量时抛出 `ValueError`
    - 测试 `debug=True` 模式不抛出异常
    - 测试 16 比特和 32 比特具体示例的往返一致性（BER=0）
    - _需求: 2.5, 6.4, 8.1, 8.2_

- [ ] 9. 最终检查点 - 确保所有测试通过
  - 确保所有测试通过，如有疑问请向用户确认。

## 备注

- 标记 `*` 的任务为可选任务，可跳过以加速 MVP 开发
- 每个任务引用了具体的需求编号以确保可追溯性
- 属性测试验证设计文档中定义的 11 个正确性属性
- 单元测试覆盖错误处理、边界情况和具体示例
- 检查点确保增量验证
- `src/utils.py` 不做修改，OSS 通过 import 复用现有接口
