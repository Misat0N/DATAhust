# 最终模型优化报告

## 当前结论

- 当前项目已切换为 `4_class_merge_23` 风险标签口径：`正常类 / 关注类 / 次级/可疑类 / 损失类`
- 已新增两阶段架构 [TwoStageCreditRiskModel](file:///Users/bytedance/scholar/DataDig/src/models/two_stage_risk_model.py)
- 已新增端到端两阶段管线 [pipeline.py](file:///Users/bytedance/scholar/DataDig/src/pipeline.py)
- 已新增终极对比实验 notebook [10_two_stage_model_optimization.ipynb](file:///Users/bytedance/scholar/DataDig/notebooks/10_two_stage_model_optimization.ipynb)

## 已验证内容

- 已完成 `23` 合并方案的代码级切换
- 当前正式方案定义为：
  - `0: 正常类`
  - `1: 关注类`
  - `2: 次级/可疑类`
  - `3: 损失类`
- 两阶段模型已完成小样本 smoke test：
-  - 输出标准 `4` 类预测标签
-  - 输出 `n x 4` 概率矩阵
  - 第一阶段阈值寻优可运行
  - 整体评估与阶段评估可运行

## 架构说明

- 第一阶段：二分类，判断 `正常类(0)` 与 `风险类(1-3 合并)`
- 第二阶段：仅对第一阶段识别出的风险样本做 `1-3` 细分
- 推理阶段继续叠加：
  - 类别权重优化
  - 正常类阈值控制
  - 业务规则兜底
  - 额度测算约束
  - LLM 策略生成

## 待正式产出

- 运行 [10_two_stage_model_optimization.ipynb](file:///Users/bytedance/scholar/DataDig/notebooks/10_two_stage_model_optimization.ipynb) 后，将正式输出：
  - `results/figures/final_model_comparison_radar.png`
  - `src/models/two_stage_risk_model.joblib`
  - 覆盖本文件中的最终指标结论

## 当前判断

- 从架构角度看，两阶段方案已经具备落地条件
- 当前最核心的不平衡矛盾在于：
  - 类 `0` 极大
  - 原始 `类3` 极小，已并入 `类2`
  - 单阶段模型仍然容易把大量样本压回 `0`
- 因此，`23` 合并后的 `4` 类两阶段方案比继续坚持原始 `5` 类更值得优先验证
