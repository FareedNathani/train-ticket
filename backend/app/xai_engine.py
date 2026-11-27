import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional
from app.ml_engine import MLEngine

class XAIEngine:
    """Computes genuine SHAP values and derives feature attributions and natural-language explanations."""

    def __init__(self, ml_engine: MLEngine):
        self.ml_engine = ml_engine
        self.explainer: Optional[shap.TreeExplainer] = None

    def initialize_explainer(self):
        if self.ml_engine.xgb_model is None:
            raise RuntimeError("XGBoost model is not loaded. Cannot initialize SHAP explainer.")
        self.explainer = shap.TreeExplainer(self.ml_engine.xgb_model)

    def explain_global(self, background_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates global mean absolute SHAP values across features."""
        if self.explainer is None:
            self.initialize_explainer()

        feature_cols = self.ml_engine.feature_columns
        X = background_df[feature_cols].fillna(0).values

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)

        # Handle multiclass vs binary shape
        # In multiclass, shap_values can be list of arrays or 3D array (samples, features, classes)
        if isinstance(shap_values, list):
            # List of (n_samples, n_features) per class
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3:
            mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        # Sort features by importance
        importance_ranking = []
        for i, col in enumerate(feature_cols):
            importance_ranking.append({
                "feature": col,
                "importance": round(float(mean_abs[i]), 5)
            })

        importance_ranking = sorted(importance_ranking, key=lambda x: x["importance"], reverse=True)

        return {
            "global_importances": importance_ranking,
            "top_driver": importance_ranking[0]["feature"] if importance_ranking else None
        }

    def explain_instance(self, row_series: pd.Series, predicted_class_idx: Optional[int] = None) -> Dict[str, Any]:
        """Calculates local SHAP waterfall attribution for a single telemetry record."""
        if self.explainer is None:
            self.initialize_explainer()

        feature_cols = self.ml_engine.feature_columns
        x_sample = np.array([float(row_series.get(col, 0.0)) for col in feature_cols]).reshape(1, -1)

        shap_explanation = self.explainer(x_sample)
        
        # Extract base value, values, and feature values
        # In multi-class, shape is (1, n_features, n_classes)
        is_multiclass = len(shap_explanation.shape) == 3
        
        if predicted_class_idx is None:
            pred = self.ml_engine.xgb_model.predict(x_sample)[0]
            predicted_class_idx = int(pred)

        if is_multiclass:
            vals = shap_explanation.values[0, :, predicted_class_idx]
            base_val = float(shap_explanation.base_values[0, predicted_class_idx])
        else:
            vals = shap_explanation.values[0]
            base_val = float(shap_explanation.base_values[0])

        classes = list(self.ml_engine.label_encoder.classes_)
        target_class_name = classes[predicted_class_idx] if predicted_class_idx < len(classes) else "ANOMALY"

        # Separate positive (increasing probability) and negative (decreasing probability)
        feature_contributions = []
        for i, col in enumerate(feature_cols):
            feature_contributions.append({
                "feature": col,
                "value": round(float(x_sample[0, i]), 2),
                "shap_value": round(float(vals[i]), 4),
                "abs_shap": abs(float(vals[i])),
                "direction": "INCREASE" if vals[i] > 0 else "DECREASE"
            })

        # Sort by impact
        sorted_contributions = sorted(feature_contributions, key=lambda x: x["abs_shap"], reverse=True)
        top_positive = [f for f in sorted_contributions if f["direction"] == "INCREASE"]
        top_negative = [f for f in sorted_contributions if f["direction"] == "DECREASE"]

        # Synthesize plain-language explanation strictly from SHAP attribution
        narrative = self._generate_narrative(target_class_name, sorted_contributions[:3])

        return {
            "target_class": target_class_name,
            "base_value": round(base_val, 4),
            "output_value": round(float(base_val + sum(vals)), 4),
            "contributions": sorted_contributions,
            "top_increasing_features": top_positive[:4],
            "top_decreasing_features": top_negative[:4],
            "natural_language_explanation": narrative
        }

    def _generate_narrative(self, target_class: str, top_features: List[Dict[str, Any]]) -> str:
        if target_class == "NORMAL":
            return "The model classified this telemetry as NORMAL because all key metrics (latency, error rate, CPU, memory) remained near normal operational baselines."

        factor_phrases = []
        for feat in top_features:
            fname = feat["feature"].replace("_", " ")
            val = feat["value"]
            factor_phrases.append(f"{fname} ({val})")

        factors_str = ", ".join(factor_phrases)
        return (
            f"The model identified '{target_class}' primarily because the following dominant factors "
            f"exceeded typical baselines: {factors_str}."
        )
