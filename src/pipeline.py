"""End-to-end credit risk pipeline for demo and batch inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from data_loader import download_dataset, parse_issue_d
from features import CreditDataPreprocessor, CreditFeatureSelector
from label_builder import enrich_labels
from llm import CreditStrategyGenerator
from models import CreditModelEvaluator, CreditRiskClassifier, CreditScorer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "src" / "models"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

RISK_LABEL_NAME_MAP = {
    0: "正常类",
    1: "关注类",
    2: "次级类",
    3: "可疑类",
    4: "损失类",
}

SCENARIO_NAME_MAP = {
    0: "正常维护",
    1: "风险预警",
    2: "逾期催收",
    3: "协商还款",
}


class CreditRiskPipeline:
    """Full-chain pipeline for risk grading, credit scoring, and strategy output."""

    def __init__(self) -> None:
        """Load existing artifacts or bootstrap them when absent."""

        self.progress_callback: Callable[[float], None] | None = None
        self.ARTIFACT_PATHS = {
            "preprocessor": ARTIFACT_DIR / "preprocessor.joblib",
            "selector": ARTIFACT_DIR / "selector.joblib",
            "risk_classifier": ARTIFACT_DIR / "risk_classifier.joblib",
            "credit_scorer": ARTIFACT_DIR / "credit_scorer.joblib",
            "metrics": ARTIFACT_DIR / "dashboard_metrics.json",
            "feature_importance": ARTIFACT_DIR / "feature_importance.csv",
        }

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        self._ensure_processed_data()
        self._ensure_artifacts()
        self._load_artifacts()

    def run_single(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Process one user end-to-end and return a structured result."""

        input_frame = self._prepare_input_frame(user_data)
        transformed_frame = self.preprocessor.transform(input_frame)
        selected_frame = self.selector.transform(transformed_frame)

        risk_label = int(self.risk_classifier.predict(selected_frame)[0])
        risk_probabilities = self.risk_classifier.predict_proba(selected_frame)[0]
        risk_probability_map = self._build_probability_map(risk_probabilities)

        scorer_input = selected_frame.copy()
        scorer_input["annual_inc"] = pd.to_numeric(
            input_frame["annual_inc"],
            errors="coerce",
        ).fillna(0.0).to_numpy()
        scorer_input["dti"] = pd.to_numeric(
            input_frame["dti"],
            errors="coerce",
        ).fillna(0.0).to_numpy()

        raw_limit = float(self.credit_scorer.predict(selected_frame)[0])
        constrained_limit = float(
            self.credit_scorer.predict_with_constraints(
                scorer_input,
                np.array([risk_label]),
            )[0]
        )

        strategy_result = self._generate_strategy(
            {
                **user_data,
                "preloan_risk_label": risk_label,
                "preloan_risk_label_name": RISK_LABEL_NAME_MAP.get(risk_label, "未知"),
                "postloan_scene_label": self._map_risk_to_scene_label(risk_label),
                "postloan_scene_label_name": SCENARIO_NAME_MAP.get(
                    self._map_risk_to_scene_label(risk_label),
                    "风险预警",
                ),
                "approved_credit_limit": constrained_limit,
            }
        )

        return {
            "risk_label": risk_label,
            "risk_label_name": RISK_LABEL_NAME_MAP.get(risk_label, "未知"),
            "risk_probabilities": risk_probability_map,
            "raw_credit_limit": round(max(raw_limit, 0.0), 2),
            "approved_credit_limit": round(max(constrained_limit, 0.0), 2),
            "strategy_scenario": strategy_result["scenario"],
            "strategy_content": strategy_result["content"],
            "strategy_source": strategy_result["source"],
        }

    def run_batch(self, batch_data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        """Process a batch of users and return a result DataFrame."""

        if isinstance(batch_data, pd.DataFrame):
            batch_frame = batch_data.copy()
        else:
            batch_frame = pd.DataFrame(batch_data)

        if batch_frame.empty:
            return pd.DataFrame()

        results = []
        total_rows = len(batch_frame)
        for row_index, (_, row) in enumerate(batch_frame.iterrows(), start=1):
            single_result = self.run_single(row.to_dict())
            results.append({**row.to_dict(), **single_result})

            if self.progress_callback is not None:
                self.progress_callback(row_index / total_rows)

        return pd.DataFrame(results)

    def _ensure_processed_data(self) -> None:
        """Create demo-friendly processed train/val/test files when missing."""

        required_paths = [
            PROCESSED_DIR / "train.csv",
            PROCESSED_DIR / "val.csv",
            PROCESSED_DIR / "test.csv",
        ]
        if all(path.exists() for path in required_paths):
            return

        self._bootstrap_processed_data(max_rows=12_000)

    def _bootstrap_processed_data(self, max_rows: int = 12_000) -> None:
        """Create lightweight processed splits from the source dataset."""

        source_csv_path = download_dataset()
        demo_frame = pd.read_csv(source_csv_path, nrows=max_rows, low_memory=False)
        demo_frame = demo_frame.drop(columns=["id", "member_id", "url"], errors="ignore")

        demo_frame["issue_d_dt"] = parse_issue_d(demo_frame["issue_d"])
        demo_frame = demo_frame.loc[demo_frame["issue_d_dt"].notna()].copy()
        demo_frame = demo_frame.sort_values("issue_d_dt", ascending=True).reset_index(drop=True)
        demo_frame = demo_frame.drop(columns=["issue_d_dt"])
        demo_frame = enrich_labels(demo_frame)

        train_end = int(len(demo_frame) * 0.70)
        val_end = int(len(demo_frame) * 0.85)

        split_frames = {
            "train": demo_frame.iloc[:train_end].copy(),
            "val": demo_frame.iloc[train_end:val_end].copy(),
            "test": demo_frame.iloc[val_end:].copy(),
        }
        for split_name, split_frame in split_frames.items():
            split_frame.to_csv(PROCESSED_DIR / f"{split_name}.csv", index=False)

    def _ensure_artifacts(self) -> None:
        """Train and cache artifacts if any required model artifact is missing."""

        if all(path.exists() for path in self.ARTIFACT_PATHS.values()):
            return

        train_df = pd.read_csv(PROCESSED_DIR / "train.csv", low_memory=False)
        val_df = pd.read_csv(PROCESSED_DIR / "val.csv", low_memory=False)
        test_df = pd.read_csv(PROCESSED_DIR / "test.csv", low_memory=False)

        preprocessor = CreditDataPreprocessor(target_column="preloan_risk_label")
        X_train_processed = preprocessor.fit_transform(train_df)
        X_val_processed = preprocessor.transform(val_df)
        X_test_processed = preprocessor.transform(test_df)

        selector = CreditFeatureSelector(
            top_k_features=30,
            use_pca=False,
            n_estimators=50,
        )
        y_train_risk = train_df["preloan_risk_label"].astype(int)
        y_val_risk = val_df["preloan_risk_label"].astype(int)
        y_test_risk = test_df["preloan_risk_label"].astype(int)

        X_train_selected = selector.fit_transform(X_train_processed, y_train_risk)
        X_val_selected = selector.transform(X_val_processed)
        X_test_selected = selector.transform(X_test_processed)

        risk_classifier = CreditRiskClassifier(model_type="lightgbm")
        risk_classifier.fit(X_train_selected, y_train_risk)

        credit_scorer = CreditScorer(model_type="lightgbm")
        y_train_limit = train_df["calibrated_credit_limit"].astype(float)
        y_test_limit = test_df["calibrated_credit_limit"].astype(float)
        credit_scorer.fit(X_train_selected, y_train_limit)

        risk_metrics = CreditModelEvaluator(
            model=risk_classifier,
            X_test=X_test_selected,
            y_test=y_test_risk,
        ).evaluate_classification()
        regression_metrics = CreditModelEvaluator(
            model=credit_scorer,
            X_test=X_test_selected,
            y_test=y_test_limit,
        ).evaluate_regression()

        metrics_payload = {
            "weighted_f1": risk_metrics["f1_weighted"],
            "ks_value": risk_metrics["ks_value"],
            "r2": regression_metrics["r2"],
            "accuracy": risk_metrics["accuracy"],
            "rmse": regression_metrics["rmse"],
        }
        self._save_feature_importance(X_train_selected, risk_classifier, selector)

        joblib.dump(preprocessor, self.ARTIFACT_PATHS["preprocessor"])
        joblib.dump(selector, self.ARTIFACT_PATHS["selector"])
        joblib.dump(risk_classifier, self.ARTIFACT_PATHS["risk_classifier"])
        joblib.dump(credit_scorer, self.ARTIFACT_PATHS["credit_scorer"])
        self.ARTIFACT_PATHS["metrics"].write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_artifacts(self) -> None:
        """Load serialized pipeline artifacts from disk."""

        self.preprocessor: CreditDataPreprocessor = joblib.load(
            self.ARTIFACT_PATHS["preprocessor"]
        )
        self.selector: CreditFeatureSelector = joblib.load(
            self.ARTIFACT_PATHS["selector"]
        )
        self.risk_classifier: CreditRiskClassifier = joblib.load(
            self.ARTIFACT_PATHS["risk_classifier"]
        )
        self.credit_scorer: CreditScorer = joblib.load(
            self.ARTIFACT_PATHS["credit_scorer"]
        )
        self.dashboard_metrics = json.loads(
            self.ARTIFACT_PATHS["metrics"].read_text(encoding="utf-8")
        )
        self.feature_importance_df = pd.read_csv(
            self.ARTIFACT_PATHS["feature_importance"]
        )

    def _save_feature_importance(
        self,
        selected_frame: pd.DataFrame,
        risk_classifier: CreditRiskClassifier,
        selector: CreditFeatureSelector,
    ) -> None:
        """Save feature importance ranking for dashboard visualization."""

        estimator = risk_classifier.best_model_
        if estimator is not None and hasattr(estimator, "feature_importances_"):
            importance_values = np.asarray(estimator.feature_importances_, dtype=float)
            importance_frame = pd.DataFrame(
                {
                    "feature": selected_frame.columns,
                    "importance": importance_values,
                }
            )
        elif selector.random_forest_ is not None:
            importance_frame = pd.DataFrame(
                {
                    "feature": selector.selected_feature_names_,
                    "importance": selector.random_forest_.feature_importances_,
                }
            )
        else:
            importance_frame = pd.DataFrame(
                {
                    "feature": selected_frame.columns,
                    "importance": np.ones(selected_frame.shape[1]),
                }
            )

        importance_frame = importance_frame.sort_values("importance", ascending=False)
        importance_frame.to_csv(self.ARTIFACT_PATHS["feature_importance"], index=False)

    def _prepare_input_frame(self, user_data: dict[str, Any]) -> pd.DataFrame:
        """Convert a single user payload into a model-ready raw DataFrame."""

        default_payload = {
            "loan_amnt": np.nan,
            "annual_inc": np.nan,
            "dti": np.nan,
            "emp_length": np.nan,
            "home_ownership": np.nan,
            "purpose": np.nan,
            "grade": np.nan,
            "sub_grade": np.nan,
            "verification_status": np.nan,
            "term": np.nan,
            "issue_d": np.nan,
            "loan_status": np.nan,
        }
        payload = {**default_payload, **user_data}
        return pd.DataFrame([payload])

    def _build_probability_map(self, probabilities: np.ndarray) -> dict[str, float]:
        """Map class probabilities to readable labels."""

        probability_map = {}
        for class_label, probability in zip(self.risk_classifier.classes_, probabilities):
            probability_map[RISK_LABEL_NAME_MAP.get(int(class_label), str(class_label))] = round(
                float(probability),
                4,
            )
        return probability_map

    def _map_risk_to_scene_label(self, risk_label: int) -> int:
        """Map a pre-loan risk label to a post-loan scenario label."""

        mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 3,
        }
        return mapping.get(int(risk_label), 1)

    def _generate_strategy(self, user_info: dict[str, Any]) -> dict[str, str]:
        """Generate a strategy via LLM, with a deterministic fallback."""

        try:
            generator = CreditStrategyGenerator()
            generated = generator.generate_strategy(user_info)
            return {
                "scenario": generated["scenario"],
                "content": generated["final_content"],
                "source": "deepseek",
            }
        except Exception:
            scenario = SCENARIO_NAME_MAP.get(
                self._map_risk_to_scene_label(int(user_info.get("preloan_risk_label", 0))),
                "风险预警",
            )
            return {
                "scenario": scenario,
                "content": self._build_fallback_strategy(user_info, scenario),
                "source": "local_fallback",
            }

    def _build_fallback_strategy(self, user_info: dict[str, Any], scenario: str) -> str:
        """Build a deterministic local fallback strategy when LLM is unavailable."""

        approved_limit = user_info.get("approved_credit_limit", "待定")
        templates = {
            "正常维护": (
                f"复贷建议：结合当前授信额度 {approved_limit} 元，优先推荐低风险复贷产品。\n"
                "权益方案：提供账单提醒、费率优惠和优质客户关怀服务。\n"
                "维护话术：感谢客户保持良好还款记录，鼓励持续稳定用信。\n"
                "合规提醒：禁止虚假承诺，沟通内容需真实、审慎、可审计。"
            ),
            "风险预警": (
                "还款提醒：提前提示账单日和还款日，建议设置自动提醒。\n"
                "风险提示：关注收入波动和负债压力变化，避免进一步逾期。\n"
                "还款规划：建议分配月度现金流，优先保障核心还款。\n"
                "合规提醒：禁止威胁辱骂，沟通必须克制、专业、真实。"
            ),
            "逾期催收": (
                "合规催收话术：提示客户当前存在逾期，请尽快核实并安排还款。\n"
                "还款方案：建议一次性还清或尽快制定阶段性还款安排。\n"
                "逾期后果：说明逾期可能影响征信和后续授信资格。\n"
                "合规提醒：禁止暴力催收、禁止威胁辱骂、禁止虚假承诺。"
            ),
            "协商还款": (
                "协商方案：优先评估客户还款意愿和可承受能力，制定分期方案。\n"
                "分期建议：按月度现金流设置合理期数，减少一次性压力。\n"
                "沟通话术：强调协商目标是解决问题并控制风险，保持尊重和克制。\n"
                "合规提醒：禁止暴力催收、禁止威胁辱骂、禁止虚假承诺。"
            ),
        }
        return templates[scenario]
