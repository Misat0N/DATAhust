# DataDig

一个面向金融风控场景的数据处理、特征工程、建模、LLM 策略生成与演示展示的 Python 项目。

## 核心功能

- 数据集分层管理：区分原始数据、预处理数据和模型工件
- 模块化代码组织：覆盖数据处理、特征工程、模型训练、LLM 与 Demo
- 风控建模链路：支持风险分级、额度测算、评估分析与可视化
- 策略生成能力：基于 Deepseek 生成贷后运营与合规话术
- 演示应用支持：提供可直接运行的 Streamlit 全链路 Demo

## 当前正式口径

- 当前贷前风险标签采用 `4_class_merge_23` 方案
- 标签定义为：`正常类 / 关注类 / 次级/可疑类 / 损失类`
- 原始 `5` 类中的 `次级类` 与 `可疑类` 已合并，原因是原 `可疑类` 样本极少，不适合继续作为独立建模类别

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

## 文档索引

- 运行指引：[RUN_GUIDE.md](file:///Users/bytedance/scholar/DataDig/docs/RUN_GUIDE.md)
- 团队贡献说明：[CONTRIBUTION.md](file:///Users/bytedance/scholar/DataDig/docs/CONTRIBUTION.md)
- 课程论文大纲：[PAPER_OUTLINE.md](file:///Users/bytedance/scholar/DataDig/docs/PAPER_OUTLINE.md)
- 演示 PPT 大纲：[PPT_OUTLINE.md](file:///Users/bytedance/scholar/DataDig/docs/PPT_OUTLINE.md)

## 全量复现说明

### 推荐环境

- Python：`3.9+`
- 虚拟环境：`venv`
- 包管理：`pip`
- 不使用：`Anaconda / conda`

### 一次性安装步骤

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows：

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### 从数据到 Demo 的完整复现顺序

1. 下载与切分数据

```bash
python3 src/data_loader.py
```

2. 构建标签

```bash
python3 src/label_builder.py
```

3. 运行 Notebook

- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_feature_engineering.ipynb`
- `notebooks/03_risk_classification.ipynb`
- `notebooks/04_credit_scoring.ipynb`
- `notebooks/05_llm_strategy_generation.ipynb`

4. 启动 Demo

```bash
bash run_demo.sh
```

或：

```bash
python3 -m streamlit run src/demo/app.py
```

### 复现完成后的主要产物

- 处理后数据：`data/processed/`
- 模型缓存工件：`src/models/artifacts/`
- 图表结果：`results/figures/`
- LLM 测试结果：`docs/`
- 演示页面：`src/demo/app.py`

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

## 测试与结果说明

### `results/` 中的结果是否已经生成

- `results/figures/` 下当前文件是已经实际运行后生成的结果，不是占位模板。
- 当前已存在的文件包括：
- `risk_label_distribution.png`
- `continuous_variable_distributions.png`
- `risk_model_confusion_matrix.png`
- `risk_model_roc_curves.png`
- `risk_model_pr_curves.png`
- `credit_scoring_prediction_vs_actual.png`
- `credit_scoring_error_distribution.png`
- `shap_summary_plot.png`
- `shap_force_plot.html`
- `shap_feature_importance.csv`
- 这些结果说明对应模块至少完成过一次实际运行或联调验证。
- 但这不等同于“所有 Notebook 都已做过一次全量正式跑数”。
- 涉及 Deepseek 在线调用的真实结果，仍需要你配置有效 `.env` 后再运行对应 Notebook。

### 各阶段测试说明

#### `01_data_exploration.ipynb`

- 用于数据探索测试
- 已覆盖数据集基本信息统计
- 已生成风险标签分布图，当前正式口径为 `4_class_merge_23`
- 已生成核心连续变量分布图
- 主要结果输出到 `results/figures/`

#### `02_feature_engineering.ipynb`

- 用于特征工程测试
- 已验证预处理器、特征选择器、采样器联调流程
- 已验证训练集拟合、验证集与测试集仅转换的无泄露逻辑
- 完整运行后会在 `data/processed/` 下生成 `.npy` 特征文件

#### `03_risk_classification.ipynb`

- 用于风险分类模型测试
- 已验证逻辑回归、决策树、随机森林、XGBoost、LightGBM 的训练流程
- 已验证最优模型调参与测试集评估流程
- 已生成混淆矩阵、ROC 曲线、PR 曲线
- 已生成 SHAP 可解释性相关结果

#### `04_credit_scoring.ipynb`

- 用于额度测算模型测试
- 已验证线性回归、Ridge、Lasso、LightGBM 的训练流程
- 已验证监管硬约束逻辑
- 已生成预测值对比图和误差分布图
- 已验证高风险用户额度归零与额度上限约束

#### `05_llm_strategy_generation.ipynb`

- 用于 LLM 策略生成测试
- 已完成代码实现和无网络假客户端联调
- 已验证场景匹配、提示词构建、合规复核与自动修正逻辑
- 配置真实 `DEEPSEEK_API_KEY` 后，可运行真实在线生成
- 运行完成后结果会输出到 `docs/`

#### `src/demo/app.py`

- 用于全链路 Demo 测试
- 已验证 `CreditRiskPipeline` 的单用户端到端流程
- 已验证 Streamlit 服务可正常启动
- 已验证 Demo 首页本地 HTTP 返回 `200`
- 未配置真实 Deepseek Key 时，策略模块会自动降级为本地模板

### 推荐复现顺序

1. 先准备数据并运行数据处理脚本
2. 运行 `01_data_exploration.ipynb`
3. 运行 `02_feature_engineering.ipynb`
4. 运行 `03_risk_classification.ipynb`
5. 运行 `04_credit_scoring.ipynb`
6. 配置 `.env` 后运行 `05_llm_strategy_generation.ipynb`
7. 最后启动 Demo：`bash run_demo.sh` 或 `run_demo.bat`

## 开发说明

- Notebook 用于展示分阶段实验流程
- `src/pipeline.py` 可作为后续 API 服务或 Web 服务的统一入口
- `src/models/artifacts/` 下保存了 Demo 运行所需的缓存工件
- `docs/` 下已补充课程提交所需运行说明、贡献说明、论文大纲和 PPT 大纲
