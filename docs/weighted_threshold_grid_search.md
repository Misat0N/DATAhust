# 权重阈值网格搜索报告

## 排序规则
- 按 `macro_f1`、`high_risk_recall`、`normal_precision` 依次降序排序。

## Top 15 配置
```text
      model_name  macro_f1  weighted_f1  ks_value  normal_precision  high_risk_recall                               config_key class_weight_name                              class_weight threshold_strategy threshold_candidate  normal_threshold  use_business_rules val_normal_precision val_high_risk_recall val_macro_f1 val_weighted_f1
权重阈值网格:mild@0.50  0.387503     0.479375  0.129097          0.992429          0.666877 mild__manual_fixed__0.50__0.50__rules_on              mild {"0": 1.0, "1": 2.0, "2": 6.0, "3": 10.0}       manual_fixed                0.50               0.5                True                 None                 None         None            None
```

## 文件输出
- `src/models/weighted_threshold_grid_search.csv`
- `src/models/weighted_threshold_grid_search.json`