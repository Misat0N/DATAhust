# DataDig

一个面向数据分析、特征工程、建模与 LLM 能力集成的 Python 项目初始化模板。

## 核心功能

- 数据集分层管理：区分原始数据与预处理数据
- 模块化代码组织：涵盖模型、特征工程、LLM 与演示应用
- 可视化结果沉淀：统一保存实验图表与分析输出
- 文档与实验记录支持：便于维护说明文档与 Notebook

## 运行指引

1. 创建并激活 Python 虚拟环境。
2. 安装依赖：`pip install -r requirements.txt`。
3. 复制环境变量模板：`cp .env.example .env`，并填写 API Key。
4. 将原始数据放入 `data/raw/`，在 `src/` 下开始开发项目代码。
5. 如需运行演示应用，可在 `src/demo/` 中编写并启动 Streamlit 脚本。
