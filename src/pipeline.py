"""End-to-end credit risk pipeline for demo and batch inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from data_loader import download_dataset, parse_issue_d
from features import CreditDataPreprocessor, CreditFeatureSelector, CreditSampler
from label_builder import enrich_labels
from llm import CreditStrategyGenerator
from models import (
    CreditModelEvaluator,
    CreditRiskClassifier,
    CreditScorer,
    RiskThresholdAdjuster,
    TwoStageCreditRiskModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "src" / "models"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FEATURE_POLICY_VERSION = "preloan_feature_policy_v5_four_class_merge23_risk"

FOUR_CLASS_RISK_LABEL_NAME_MAP = {
    0: "正常类",
    1: "关注类",
    2: "次级/可疑类",
    3: "损失类",
}


def get_preferred_processed_split_path(split_name: str) -> Path:
    """Return the purified split when available, otherwise fall back to default."""

    purified_path = PROCESSED_DIR / f"purified_{split_name}.csv"
    if purified_path.exists():
        return purified_path
    return PROCESSED_DIR / f"{split_name}.csv"

THREE_CLASS_RISK_LABEL_NAME_MAP = {
    0: "正常类",
    1: "风险类",
    2: "损失类",
}

SCENARIO_NAME_MAP = {
    0: "正常维护",
    1: "风险预警",
    2: "逾期催收",
    3: "协商还款",
}


class CreditRiskPipeline:
    """Full-chain pipeline for risk grading, credit scoring, and strategy output."""

    def __init__(self, use_sampling_optimization: bool = False) -> None:
        """Load existing artifacts or bootstrap them when absent."""

        self.progress_callback: Callable[[float], None] | None = None
        self.use_sampling_optimization = use_sampling_optimization
        self.ARTIFACT_PATHS = {
            "preprocessor": ARTIFACT_DIR / "preprocessor.joblib",
            "selector": ARTIFACT_DIR / "selector.joblib",
            "risk_classifier": ARTIFACT_DIR / "risk_classifier.joblib",
            "credit_scorer": ARTIFACT_DIR / "credit_scorer.joblib",
            "metrics": ARTIFACT_DIR / "dashboard_metrics.json",
            "feature_importance": ARTIFACT_DIR / "feature_importance.csv",
            "feature_policy": ARTIFACT_DIR / "feature_policy.json",
            "threshold_config": MODEL_DIR / "best_threshold_config.json",
        }

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        self._ensure_processed_data()
        self._ensure_artifacts()
        self._load_artifacts()
        self.threshold_adjuster = RiskThresholdAdjuster(
            normal_class_threshold=self._load_saved_threshold()
        )

    def set_threshold(self, normal_class_threshold: float) -> None:
        """Update the normal-class threshold used by the adjuster."""

        self.threshold_adjuster = RiskThresholdAdjuster(
            normal_class_threshold=normal_class_threshold
        )

    def set_sampling_enabled(self, enabled: bool) -> None:
        """Enable or disable sampling optimization for future training runs."""

        self.use_sampling_optimization = bool(enabled)

    def run_single(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Process one user end-to-end and return a structured result."""

        input_frame = self._prepare_input_frame(user_data)
        transformed_frame = self.preprocessor.transform(input_frame)
        selected_frame = self.selector.transform(transformed_frame)

        raw_model_risk_label = int(self.risk_classifier.predict(selected_frame)[0])
        risk_probabilities = self.risk_classifier.predict_proba(selected_frame)[0]
        threshold_adjusted_risk_label = int(
            self.threshold_adjuster.adjust_prediction(
                np.asarray([risk_probabilities], dtype=float)
            )[0]
        )
        final_risk_label = int(
            self.threshold_adjuster.business_rule_override(
                input_frame.iloc[0],
                threshold_adjusted_risk_label,
            )
        )
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
        scorer_risk_label = self._map_risk_to_limit_constraint_label(final_risk_label)
        constrained_limit = float(
            self.credit_scorer.predict_with_constraints(
                scorer_input,
                np.array([scorer_risk_label]),
            )[0]
        )

        strategy_result = self._generate_strategy(
            {
                **user_data,
                "raw_model_risk_label": raw_model_risk_label,
                "raw_model_risk_label_name": self._get_risk_label_name(
                    raw_model_risk_label
                ),
                "threshold_adjusted_risk_label": threshold_adjusted_risk_label,
                "threshold_adjusted_risk_label_name": self._get_risk_label_name(
                    threshold_adjusted_risk_label
                ),
                "preloan_risk_label": final_risk_label,
                "preloan_risk_label_name": self._get_risk_label_name(final_risk_label),
                "postloan_scene_label": self._map_risk_to_scene_label(final_risk_label),
                "postloan_scene_label_name": SCENARIO_NAME_MAP.get(
                    self._map_risk_to_scene_label(final_risk_label),
                    "风险预警",
                ),
                "approved_credit_limit": constrained_limit,
            }
        )

        return {
            "raw_model_risk_label": raw_model_risk_label,
            "raw_model_risk_label_name": self._get_risk_label_name(
                raw_model_risk_label
            ),
            "threshold_adjusted_risk_label": threshold_adjusted_risk_label,
            "threshold_adjusted_risk_label_name": self._get_risk_label_name(
                threshold_adjusted_risk_label
            ),
            "rule_adjusted_risk_label": final_risk_label,
            "rule_adjusted_risk_label_name": self._get_risk_label_name(
                final_risk_label
            ),
            "risk_label": final_risk_label,
            "risk_label_name": self._get_risk_label_name(final_risk_label),
            "risk_probabilities": risk_probability_map,
            "normal_class_threshold": self.threshold_adjuster.normal_class_threshold,
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
            feature_policy = json.loads(
                self.ARTIFACT_PATHS["feature_policy"].read_text(encoding="utf-8")
            )
            if feature_policy.get("version") == FEATURE_POLICY_VERSION:
                return

        train_df = pd.read_csv(
            get_preferred_processed_split_path("train"),
            low_memory=False,
        )
        test_df = pd.read_csv(
            get_preferred_processed_split_path("test"),
            low_memory=False,
        )

        preprocessor = CreditDataPreprocessor(target_column="preloan_risk_label")
        X_train_processed = preprocessor.fit_transform(train_df)
        X_test_processed = preprocessor.transform(test_df)

        selector = CreditFeatureSelector(
            top_k_features=30,
            use_pca=False,
            n_estimators=50,
        )
        y_train_risk = train_df["preloan_risk_label"].astype(int)
        y_test_risk = test_df["preloan_risk_label"].astype(int)

        if self.use_sampling_optimization:
            sampler = CreditSampler(
                categorical_features=self._infer_categorical_feature_indices(
                    X_train_processed
                )
            )
            X_train_for_selector, y_train_for_selector = sampler.fit_resample(
                X_train_processed,
                y_train_risk,
            )
        else:
            X_train_for_selector = X_train_processed
            y_train_for_selector = y_train_risk

        X_train_selected = selector.fit_transform(
            X_train_for_selector,
            y_train_for_selector,
        )
        X_test_selected = selector.transform(X_test_processed)

        risk_classifier = CreditRiskClassifier(model_type="lightgbm")
        risk_classifier.fit(X_train_selected, y_train_for_selector)

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
            "risk_class_count": len(np.unique(y_test_risk)),
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
        self.ARTIFACT_PATHS["feature_policy"].write_text(
            json.dumps(
                {
                    "version": FEATURE_POLICY_VERSION,
                    "description": "Only pre-loan application and bureau fields are allowed as model features.",
                    "risk_scheme": "4_class_merge_23",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

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
            "delinq_2yrs": np.nan,
            "inq_last_6mths": np.nan,
            "delinq_amnt": np.nan,
            "installment": np.nan,
            "hardship_dpd": np.nan,
            "historical_max_overdue_days": np.nan,
            "estimated_overdue_days": np.nan,
        }
        payload = {**default_payload, **user_data}
        return pd.DataFrame([payload])

    def _infer_categorical_feature_indices(self, frame: pd.DataFrame) -> list[int]:
        """Infer one-hot categorical feature positions for SMOTENC.

        The current preprocessor emits:
        - continuous features with the suffix ``__scaled``
        - target-encoded features with the suffix ``__target_encoded``
        - one-hot categorical features using ``feature=value``

        Only the one-hot categorical columns should be marked as categorical for
        ``SMOTENC``.
        """

        categorical_indices = [
            index
            for index, column in enumerate(frame.columns)
            if "=" in column and not column.endswith("__target_encoded")
        ]
        return categorical_indices

    def _build_probability_map(self, probabilities: np.ndarray) -> dict[str, float]:
        """Map class probabilities to readable labels."""

        probability_map = {}
        for class_label, probability in zip(self.risk_classifier.classes_, probabilities):
            probability_map[self._get_risk_label_name(int(class_label))] = round(
                float(probability),
                4,
            )
        return probability_map

    def _get_risk_label_name(self, risk_label: int) -> str:
        """Resolve a readable risk label while staying backward compatible."""

        class_count = self._get_active_risk_class_count()
        if class_count <= 3 and risk_label in THREE_CLASS_RISK_LABEL_NAME_MAP:
            return THREE_CLASS_RISK_LABEL_NAME_MAP[risk_label]
        return FOUR_CLASS_RISK_LABEL_NAME_MAP.get(risk_label, "未知")

    def _map_risk_to_scene_label(self, risk_label: int) -> int:
        """Map a pre-loan risk label to a post-loan scenario label."""

        class_count = self._get_active_risk_class_count()
        if class_count <= 3:
            mapping = {
                0: 0,
                1: 1,
                2: 3,
            }
            return mapping.get(int(risk_label), 1)

        mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        }
        return mapping.get(int(risk_label), 1)

    def _map_risk_to_limit_constraint_label(self, risk_label: int) -> int:
        """Map the final risk label into the scorer's current 3-level constraints."""

        class_count = self._get_active_risk_class_count()
        if class_count <= 3:
            mapping = {
                0: 0,
                1: 1,
                2: 2,
            }
            return mapping.get(int(risk_label), 2)

        return int(np.clip(int(risk_label), 0, 3))

    def _get_active_risk_class_count(self) -> int:
        """Return the active class count for either single-stage or two-stage models."""

        if hasattr(self, "risk_classifier"):
            return len(np.asarray(getattr(self.risk_classifier, "classes_", [])))
        if hasattr(self, "two_stage_risk_model"):
            return len(np.asarray(getattr(self.two_stage_risk_model, "classes_", [])))
        return 0

    def _load_saved_threshold(self) -> float:
        """Load the saved threshold configuration when available."""

        threshold_path = self.ARTIFACT_PATHS["threshold_config"]
        if not threshold_path.exists():
            return 0.9

        try:
            payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        except Exception:
            return 0.9

        return float(payload.get("best_threshold", 0.9))


class TwoStageCreditRiskPipeline(CreditRiskPipeline):
    """End-to-end pipeline powered by the two-stage four-class risk model."""

    def __init__(self, use_sampling_optimization: bool = False) -> None:
        """Load or bootstrap artifacts for the two-stage pipeline."""

        self.progress_callback: Callable[[float], None] | None = None
        self.use_sampling_optimization = use_sampling_optimization
        self.ARTIFACT_PATHS = {
            "preprocessor": ARTIFACT_DIR / "two_stage_preprocessor.joblib",
            "selector": ARTIFACT_DIR / "two_stage_selector.joblib",
            "two_stage_risk_model": MODEL_DIR / "two_stage_risk_model.joblib",
            "credit_scorer": ARTIFACT_DIR / "two_stage_credit_scorer.joblib",
            "metrics": ARTIFACT_DIR / "two_stage_dashboard_metrics.json",
            "feature_importance": ARTIFACT_DIR / "two_stage_feature_importance.csv",
            "feature_policy": ARTIFACT_DIR / "two_stage_feature_policy.json",
            "threshold_config": MODEL_DIR / "best_threshold_config.json",
        }

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        self._ensure_processed_data()
        self._ensure_artifacts()
        self._load_artifacts()
        self.threshold_adjuster = RiskThresholdAdjuster(
            normal_class_threshold=self._load_saved_threshold()
        )

    def run_single(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Process one user with the two-stage model and business rules."""

        input_frame = self._prepare_input_frame(user_data)
        transformed_frame = self.preprocessor.transform(input_frame)
        selected_frame = self.selector.transform(transformed_frame)

        raw_model_risk_label = int(self.two_stage_risk_model.predict(selected_frame)[0])
        risk_probabilities = self.two_stage_risk_model.predict_proba(selected_frame)[0]
        threshold_adjusted_risk_label = int(
            self.threshold_adjuster.adjust_prediction(
                np.asarray([risk_probabilities], dtype=float)
            )[0]
        )
        final_risk_label = int(
            self.threshold_adjuster.business_rule_override(
                input_frame.iloc[0],
                threshold_adjusted_risk_label,
            )
        )
        risk_probability_map = self._build_two_stage_probability_map(risk_probabilities)

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
                np.array([final_risk_label]),
            )[0]
        )

        strategy_result = self._generate_strategy(
            {
                **user_data,
                "raw_model_risk_label": raw_model_risk_label,
                "raw_model_risk_label_name": self._get_risk_label_name(
                    raw_model_risk_label
                ),
                "threshold_adjusted_risk_label": threshold_adjusted_risk_label,
                "threshold_adjusted_risk_label_name": self._get_risk_label_name(
                    threshold_adjusted_risk_label
                ),
                "preloan_risk_label": final_risk_label,
                "preloan_risk_label_name": self._get_risk_label_name(final_risk_label),
                "postloan_scene_label": self._map_risk_to_scene_label(final_risk_label),
                "postloan_scene_label_name": SCENARIO_NAME_MAP.get(
                    self._map_risk_to_scene_label(final_risk_label),
                    "风险预警",
                ),
                "approved_credit_limit": constrained_limit,
            }
        )

        return {
            "raw_model_risk_label": raw_model_risk_label,
            "raw_model_risk_label_name": self._get_risk_label_name(
                raw_model_risk_label
            ),
            "threshold_adjusted_risk_label": threshold_adjusted_risk_label,
            "threshold_adjusted_risk_label_name": self._get_risk_label_name(
                threshold_adjusted_risk_label
            ),
            "rule_adjusted_risk_label": final_risk_label,
            "rule_adjusted_risk_label_name": self._get_risk_label_name(
                final_risk_label
            ),
            "risk_label": final_risk_label,
            "risk_label_name": self._get_risk_label_name(final_risk_label),
            "risk_probabilities": risk_probability_map,
            "normal_class_threshold": self.threshold_adjuster.normal_class_threshold,
            "raw_credit_limit": round(max(raw_limit, 0.0), 2),
            "approved_credit_limit": round(max(constrained_limit, 0.0), 2),
            "strategy_scenario": strategy_result["scenario"],
            "strategy_content": strategy_result["content"],
            "strategy_source": strategy_result["source"],
        }

    def _ensure_artifacts(self) -> None:
        """Train and cache the two-stage pipeline artifacts when needed."""

        if all(path.exists() for path in self.ARTIFACT_PATHS.values()):
            try:
                feature_policy = json.loads(
                    self.ARTIFACT_PATHS["feature_policy"].read_text(encoding="utf-8")
                )
                if feature_policy.get("version") == FEATURE_POLICY_VERSION:
                    return
            except Exception:
                pass

        train_df = pd.read_csv(
            get_preferred_processed_split_path("train"),
            low_memory=False,
        )
        test_df = pd.read_csv(
            get_preferred_processed_split_path("test"),
            low_memory=False,
        )

        preprocessor = CreditDataPreprocessor(target_column="preloan_risk_label")
        X_train_processed = preprocessor.fit_transform(train_df)
        X_test_processed = preprocessor.transform(test_df)

        y_train_risk = train_df["preloan_risk_label"].astype(int)
        y_test_risk = test_df["preloan_risk_label"].astype(int)

        if self.use_sampling_optimization:
            sampler = CreditSampler(
                categorical_features=self._infer_categorical_feature_indices(
                    X_train_processed
                )
            )
            X_train_for_selector, y_train_for_selector = sampler.fit_resample(
                X_train_processed,
                y_train_risk,
            )
        else:
            X_train_for_selector = X_train_processed
            y_train_for_selector = y_train_risk

        selector = CreditFeatureSelector(
            top_k_features=80,
            use_pca=False,
            n_estimators=100,
        )
        X_train_selected = selector.fit_transform(
            X_train_for_selector,
            y_train_for_selector,
        )
        X_test_selected = selector.transform(X_test_processed)
        X_train_selected_raw = selector.transform(X_train_processed)

        two_stage_model = TwoStageCreditRiskModel()
        two_stage_model.fit(X_train_selected, y_train_for_selector)

        credit_scorer = CreditScorer(model_type="lightgbm")
        y_train_limit = train_df["calibrated_credit_limit"].astype(float)
        y_test_limit = test_df["calibrated_credit_limit"].astype(float)
        credit_scorer.fit(X_train_selected_raw, y_train_limit)

        risk_metrics = two_stage_model.evaluate_two_stage_model(
            X_test_selected,
            y_test_risk,
        )["overall"]
        regression_metrics = CreditModelEvaluator(
            model=credit_scorer,
            X_test=X_test_selected,
            y_test=y_test_limit,
        ).evaluate_regression()

        metrics_payload = {
            "weighted_f1": risk_metrics["weighted_f1"],
            "macro_f1": risk_metrics["macro_f1"],
            "ks_value": risk_metrics["ks_value"],
            "r2": regression_metrics["r2"],
            "accuracy": risk_metrics["accuracy"],
            "rmse": regression_metrics["rmse"],
            "risk_class_count": len(np.unique(y_test_risk)),
            "normal_threshold": two_stage_model.get_threshold(),
        }
        self._save_two_stage_feature_importance(
            X_train_selected_raw,
            two_stage_model,
        )

        joblib.dump(preprocessor, self.ARTIFACT_PATHS["preprocessor"])
        joblib.dump(selector, self.ARTIFACT_PATHS["selector"])
        joblib.dump(two_stage_model, self.ARTIFACT_PATHS["two_stage_risk_model"])
        joblib.dump(credit_scorer, self.ARTIFACT_PATHS["credit_scorer"])
        self.ARTIFACT_PATHS["metrics"].write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.ARTIFACT_PATHS["feature_policy"].write_text(
            json.dumps(
                {
                    "version": FEATURE_POLICY_VERSION,
                    "description": "Only pre-loan application and bureau fields are allowed as model features.",
                    "risk_scheme": "4_class_merge_23",
                    "architecture": "two_stage",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_artifacts(self) -> None:
        """Load the serialized two-stage pipeline artifacts from disk."""

        self.preprocessor: CreditDataPreprocessor = joblib.load(
            self.ARTIFACT_PATHS["preprocessor"]
        )
        self.selector: CreditFeatureSelector = joblib.load(
            self.ARTIFACT_PATHS["selector"]
        )
        self.two_stage_risk_model: TwoStageCreditRiskModel = joblib.load(
            self.ARTIFACT_PATHS["two_stage_risk_model"]
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

    def _build_two_stage_probability_map(
        self,
        probabilities: np.ndarray,
    ) -> dict[str, float]:
        """Map the four-class probability vector to readable names."""

        probability_map = {}
        for class_label, probability in zip(self.two_stage_risk_model.classes_, probabilities):
            probability_map[self._get_risk_label_name(int(class_label))] = round(
                float(probability),
                4,
            )
        return probability_map

    def _save_two_stage_feature_importance(
        self,
        selected_frame: pd.DataFrame,
        two_stage_model: TwoStageCreditRiskModel,
    ) -> None:
        """Save the stage-two feature importance when available."""

        if (
            two_stage_model.second_stage_constant_label_ is None
            and two_stage_model.second_stage_model.best_model_ is not None
            and hasattr(two_stage_model.second_stage_model.best_model_, "feature_importances_")
        ):
            importance_values = np.asarray(
                two_stage_model.second_stage_model.best_model_.feature_importances_,
                dtype=float,
            )
            importance_frame = pd.DataFrame(
                {
                    "feature": selected_frame.columns,
                    "importance": importance_values,
                }
            )
        elif (
            two_stage_model.first_stage_model.best_model_ is not None
            and hasattr(two_stage_model.first_stage_model.best_model_, "feature_importances_")
        ):
            importance_values = np.asarray(
                two_stage_model.first_stage_model.best_model_.feature_importances_,
                dtype=float,
            )
            importance_frame = pd.DataFrame(
                {
                    "feature": selected_frame.columns,
                    "importance": importance_values,
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
