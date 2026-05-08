# 项目运行指引

## 1. 环境搭建步骤

### 1.1 系统要求

- 操作系统：macOS / Linux / Windows
- Python 版本：建议 `3.9` 及以上
- 包管理方式：`venv + pip`
- 不使用 `Anaconda / conda`

### 1.2 创建虚拟环境

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows：

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 1.3 安装项目依赖

```bash
pip install -r requirements.txt
```

### 1.4 安装系统依赖

如果在 macOS 上使用 `lightgbm`，建议提前安装：

```bash
brew install libomp
```

### 1.5 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

根据实际情况填写：

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`

说明：

- 若未配置真实 `DEEPSEEK_API_KEY`，LLM 模块会自动降级为本地模板策略
- 这不会影响非 LLM 模块运行

## 2. 数据集下载与预处理说明

### 2.1 数据集来源

- 项目使用 Lending Club 数据集
- 下载方式基于 `kagglehub`
- 数据下载代码位于 `src/data_loader.py`

### 2.2 下载与切分数据

运行：

```bash
python3 src/data_loader.py
```

该脚本会完成：

- 使用 `kagglehub` 下载原始数据
- 读取 CSV 并检查数据基本信息
- 剔除包含唯一标识的字段，如 `id`、`member_id`、`url`
- 基于 `issue_d` 做升序排序
- 按 `70% / 15% / 15%` 做时序切分
- 输出：
  - `data/processed/train.csv`
  - `data/processed/val.csv`
  - `data/processed/test.csv`

### 2.3 构建标签

运行：

```bash
python3 src/label_builder.py
```

该脚本会补充：

- `4_class_merge_23` 贷前风险标签
- 风险标签定义：`正常类 / 关注类 / 次级/可疑类 / 损失类`
- 校准后的授信额度标签
- 4 类贷后场景标签

输出文件会覆盖写回：

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## 3. 模型训练与评估步骤

### 3.1 数据探索

打开并运行：

- `notebooks/01_data_exploration.ipynb`

主要内容：

- 数据集基本信息统计
- 风险标签分布可视化
- 连续变量分布与箱线图分析

主要输出：

- `results/figures/risk_label_distribution.png`
- `results/figures/continuous_variable_distributions.png`

### 3.2 特征工程

打开并运行：

- `notebooks/02_feature_engineering.ipynb`

主要内容：

- 训练集拟合预处理器
- 特征筛选与可选降维
- 训练集采样处理
- 输出 `.npy` 特征文件

### 3.3 风险分类模型训练与评估

打开并运行：

- `notebooks/03_risk_classification.ipynb`

主要内容：

- 训练逻辑回归、决策树、随机森林、XGBoost、LightGBM
- 选择最优模型并做时序交叉验证调参
- 在测试集上生成分类指标和可解释性结果

说明：

- 当前正式风险标签口径为 `4_class_merge_23`
- 若文档中仍看到旧的 `5` 类表述，应以 `label_builder.py` 和最新处理后数据为准

主要输出：

- `results/figures/risk_model_confusion_matrix.png`
- `results/figures/risk_model_roc_curves.png`
- `results/figures/risk_model_pr_curves.png`
- `results/figures/shap_summary_plot.png`
- `results/figures/shap_force_plot.html`
- `results/figures/shap_feature_importance.csv`

### 3.4 额度测算模型训练与评估

打开并运行：

- `notebooks/04_credit_scoring.ipynb`

主要内容：

- 训练线性回归、Ridge、Lasso、LightGBM 回归模型
- 选择最优模型并调参
- 验证监管硬约束逻辑

主要输出：

- `results/figures/credit_scoring_prediction_vs_actual.png`
- `results/figures/credit_scoring_error_distribution.png`

### 3.5 LLM 策略生成测试

打开并运行：

- `notebooks/05_llm_strategy_generation.ipynb`

主要内容：

- 测试 Deepseek API 连接
- 对 10 条测试用户信息做策略生成
- 执行二次合规校验
- 保存测试结果到 `docs/`

注意：

- 该步骤依赖真实 `DEEPSEEK_API_KEY`
- 若只做离线演示，可使用系统本地模板兜底

## 4. Demo 启动与演示说明

### 4.1 启动方式

macOS / Linux：

```bash
bash run_demo.sh
```

Windows：

```bat
run_demo.bat
```

或者手动执行：

```bash
python3 -m streamlit run src/demo/app.py
```

### 4.2 Demo 页面结构

- 首页：
  - 项目介绍
  - 系统架构图
  - 核心功能说明
- 单用户查询页：
  - 输入核心用户特征
  - 展示风险等级、概率分布、合规额度、策略话术
- 批量处理页：
  - 上传 `CSV / Excel`
  - 返回批量结果
  - 支持导出 Excel 报告
- 可视化看板页：
  - 风险分布饼图
  - 关键指标展示
  - 特征重要性排序

### 4.3 演示建议顺序

1. 首页介绍项目背景和系统架构
2. 单用户页展示一次完整评估流程
3. 批量处理页展示上传、处理与导出
4. 可视化看板页展示模型指标和图表结果

## 5. 常见问题排查

### 5.1 `No module named ...`

原因：

- 虚拟环境未激活
- 依赖未完整安装

处理方式：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 `lightgbm` 安装失败

原因：

- 缺少系统级依赖

处理方式：

```bash
brew install libomp
pip install lightgbm
```

### 5.3 Deepseek 调用失败

排查方向：

- `.env` 是否存在
- `DEEPSEEK_API_KEY` 是否正确
- `DEEPSEEK_BASE_URL` 是否可访问
- 网络是否可连通

说明：

- 即使 Deepseek 不可用，Demo 仍可使用本地模板完成演示

### 5.4 Demo 启动后页面打不开

排查方向：

- 端口 `8501` 是否被占用
- 是否在正确目录启动
- 是否已安装 `streamlit`

可尝试：

```bash
python3 -m streamlit run src/demo/app.py --server.port 8502
```

### 5.5 批量上传失败

建议检查：

- 文件是否为 `.csv` 或 `.xlsx`
- 字段名是否包含必要列
- 数值列是否混入非法字符串

### 5.6 结果图未生成

建议检查：

- Notebook 是否完整执行
- `results/figures/` 是否有写权限
- 运行过程中是否出现异常中断

## 6. 推荐复现顺序

1. 搭建虚拟环境并安装依赖
2. 配置 `.env`
3. 运行 `src/data_loader.py`
4. 运行 `src/label_builder.py`
5. 按顺序执行 `01` 到 `05` Notebook
6. 启动 Demo 并进行演示
