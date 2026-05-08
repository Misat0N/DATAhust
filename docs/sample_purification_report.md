# 样本提纯报告

> 说明：该文件为样本提纯实验报告占位稿。运行 `notebooks/11_sample_purification.ipynb` 后，会自动覆盖为最新结果。

## 当前状态

- `CreditSamplePurifier` 已完成代码实现
- `11_sample_purification.ipynb` 已完成实验脚本搭建
- 当前脚本会对 `train / val / test` 三个数据集都执行规则提纯
- 默认仅对 `train` 额外执行分层抽样，`val / test` 保持提纯后自然分布
- 正式分布变化与模型对比指标，需运行 notebook 后写入

## 预期输出

- `data/processed/purified_train.csv`
- `data/processed/purified_val.csv`
- `data/processed/purified_test.csv`
- `results/figures/sample_purification_comparison.png`
- 原始训练集 vs 提纯后训练集的风险分类指标对比
