# DataDig

一个面向金融风控场景的数据处理、特征工程、建模、LLM 策略生成与演示展示的 Python 项目。

## 核心功能

- 数据集分层管理：区分原始数据、预处理数据和模型工件
- 模块化代码组织：覆盖数据处理、特征工程、模型训练、LLM 与 Demo
- 风控建模链路：支持风险分级、额度测算、评估分析与可视化
- 策略生成能力：基于 Deepseek 生成贷后运营与合规话术
- 演示应用支持：提供可直接运行的 Streamlit 全链路 Demo

## 环境准备

1. 创建并激活 Python 虚拟环境。
2. 安装依赖：`pip install -r requirements.txt`
3. 复制环境变量模板：

```bash
cp .env.example .env
```

4. 如需启用 LLM 策略生成，请在 `.env` 中填写真实的 `DEEPSEEK_API_KEY`
5. 如需运行 LightGBM，macOS 建议提前安装系统依赖：`brew install libomp`

## 项目结构

- `data/raw/`：原始数据
- `data/processed/`：预处理后的训练、验证、测试数据
- `src/features/`：预处理、特征筛选、采样
- `src/models/`：风险分级、额度测算、模型评估
- `src/llm/`：Deepseek 客户端与策略生成
- `src/demo/`：Streamlit 演示应用
- `src/pipeline.py`：全链路统一入口
- `results/figures/`：图表输出
- `docs/`：文档和 LLM 测试结果
- `notebooks/`：实验与演示 Notebook

## Demo 使用

### 启动方式

Mac / Linux：

```bash
bash run_demo.sh
```

Windows：

```bat
run_demo.bat
```

如果你已经手动激活虚拟环境，也可以直接执行：

```bash
python3 -m streamlit run src/demo/app.py
```

### 页面说明

- 首页：展示项目介绍、系统架构图和核心功能
- 单用户查询：输入用户核心特征，完成风险分级、额度测算和策略生成
- 批量处理：上传 `CSV` 或 `Excel`，批量生成结果并导出 Excel 报告
- 可视化看板：查看风险分布、模型指标和特征重要性

### 单用户查询使用方法

1. 打开“单用户查询”页
2. 输入年收入、负债比、申请额度、工作年限、住房情况等信息
3. 点击“开始评估”
4. 查看输出结果：
   - 风险等级及概率分布
   - 合规授信额度
   - 贷后策略与话术

### 批量处理使用方法

1. 打开“批量处理”页
2. 上传 `CSV` 或 `Excel` 文件
3. 点击“批量处理”
4. 等待进度完成后查看结果表格
5. 点击“下载 Excel 结果报告”导出结果

建议批量文件至少包含以下字段：

- `annual_inc`
- `dti`
- `loan_amnt`
- `emp_length`
- `home_ownership`
- `verification_status`
- `purpose`
- `grade`
- `term`
- `issue_d`
- `loan_status`

### LLM 策略说明

- 若 `.env` 中已配置真实 Deepseek Key，系统会优先调用 Deepseek 生成策略
- 若未配置或调用失败，系统会自动降级为本地合规模板策略
- 本地模板同样遵循基础合规约束，适合离线演示

## 开发说明

- Notebook 用于展示分阶段实验流程
- `src/pipeline.py` 可作为后续 API 服务或 Web 服务的统一入口
- `src/models/artifacts/` 下保存了 Demo 运行所需的缓存工件
