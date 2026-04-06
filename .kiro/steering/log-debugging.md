---
inclusion: manual
---

# Log 文件调试指南

## 读取策略

- 优先读取 log 文件的最后 100 行（通常包含最终的报错信息和 traceback）
- 如果末尾信息不足以定位问题，再向上扩展读取范围
- 不要一次性读取整个 log 文件，避免浪费 token

## 分析流程

1. 先从 log 末尾提取 Python traceback 或 error 信息
2. 识别报错类型（如 RuntimeError、CUDA OOM、shape mismatch 等）
3. 定位报错发生在哪个源文件的哪一行
4. 读取对应的源代码，分析根因
5. 给出具体的修复方案

## 常见报错模式

- CUDA out of memory：建议减小 batch_size 或图片分辨率，或使用 gradient checkpointing
- shape mismatch / size mismatch：检查 tensor 维度，追溯数据流
- KeyError / AttributeError：检查配置参数或模型结构是否匹配
- FileNotFoundError：检查路径和数据集是否正确配置

## 输出格式

- 先简要说明报错原因（一两句话）
- 再给出具体的修复代码或操作步骤
- 如果有多个报错，按出现顺序逐个处理
