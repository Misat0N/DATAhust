# Baseline 模型对比

## 指标说明
- `weighted_f1_score`: 加权 F1-Score
- `average_auc`: 多分类各类别 AUC 的简单平均
- `ks_value`: 各类别 KS 的平均值

## 结果表
         model_type model_name  weighted_f1_score  average_auc  ks_value
logistic_regression       逻辑回归           0.907761     0.573830  0.111549
      random_forest       随机森林           0.916268     0.589610  0.135059
            xgboost    XGBoost           0.909108     0.609233  0.164497

## 文件输出
- `src/models/baseline_model_metrics.csv`
- `src/models/baseline_model_metrics.json`