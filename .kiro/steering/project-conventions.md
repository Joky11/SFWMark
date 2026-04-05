---
inclusion: always
---

# 项目约定

## 语言要求

- 所有与用户的对话、回复、解释必须使用中文
- 代码中的注释必须使用中文
- 变量名、函数名、类名等标识符保持英文（遵循编程惯例）

## 虚拟环境

- 在执行任何命令行操作（测试、安装依赖、运行脚本等）之前，必须先激活 conda 虚拟环境
- 激活命令：`conda activate sfw`
- 每次调用 bash 命令时，都需要先运行 `conda activate sfw` 再执行目标命令
- 示例：先执行 `conda activate sfw`，再执行 `pytest ...`
