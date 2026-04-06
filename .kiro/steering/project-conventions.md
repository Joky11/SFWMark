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

## 项目路径

- 项目根目录为工作区根目录，不要使用 `cd` 切换到其他路径
- 执行 bash 命令时不要指定 `cwd` 参数，直接在工作区根目录执行即可
- 绝对不要猜测项目路径（如 `/home/user/...`），始终使用相对路径

## 测试运行

- 运行 pytest 时必须设置 `PYTHONPATH=src`，因为 `src/` 下的模块使用不带 `src.` 前缀的相对导入（如 `from utils import ...`）
- 正确的测试命令格式：`conda activate sfw && PYTHONPATH=src python -m pytest tests/ -v`
- 运行单个测试文件：`conda activate sfw && PYTHONPATH=src python -m pytest tests/test_xxx.py -v`
