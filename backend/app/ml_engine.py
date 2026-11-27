import os
import json
import joblib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import xgboost as xgb
from app.config import MODELS_DIR, ANOMALY_CATEGORIES
from app.pipeline import TelemetryFeaturePipeline

class MLEngine:
    """Manages model training, evaluation, comparison, and inference for XGBoost and Random Forest."""

    def __init__(self):
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = TelemetryFeaturePipeline.FEATURE_COLUMNS.copy()
        self.model_metadata = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        test_size: float = 0.2,
        split_method: str = "stratified"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepares feature matrix X and label vector y with stratified or temporal split."""
        # Ensure required feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_columns].fillna(0).values
        raw_y = df[label_col].astype(str).values

        # Encode labels
        y = self.label_encoder.fit_transform(raw_y)

        if split_method == "temporal":
            # Temporal split without shuffling
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            # Stratified split with fallback if class count too low
            min_class_count = min(np.bincount(y)) if len(np.unique(y)) > 1 else 1
            if min_class_count >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, shuffle=True
                )

        return X_train, X_test, y_train, y_test

    def train_models(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        test_size: float = 0.25,
        split_method: str = "stratified",
        n_estimators: int = 120,
        max_depth: int = 6,
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Trains both XGBoost and Random Forest, calculating genuine evaluation metrics."""
        if len(df) < 10:
            raise ValueError("Dataset has fewer than 10 samples. More telemetry records required.")

        X_train, X_test, y_train, y_test = self.prepare_data(
            df, label_col=label_col, test_size=test_size, split_method=split_method
        )

        num_classes = len(self.label_encoder.classes_)
        objective = "multi:softprob" if num_classes > 2 else "binary:logistic"

        # 1. Train XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
            random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
        xgb_preds = self.xgb_model.predict(X_test)
        xgb_probs = self.xgb_model.predict_proba(X_test)

        # 2. Train Random Forest (Baseline)
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42
        )
        self.rf_model.fit(X_train, y_train)
        rf_preds = self.rf_model.predict(X_test)

        # Calculate metrics
        classes = list(self.label_encoder.classes_)

        def calculate_metrics(y_true, y_pred):
            acc = float(accuracy_score(y_true, y_pred))
            prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            cm = confusion_matrix(y_true, y_pred).tolist()
            report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
            return {
                "accuracy": round(acc, 4),
                "precision": round(prec_macro, 4),
                "recall": round(rec_macro, 4),
                "f1_score": round(f1_macro, 4),
                "f1_weighted": round(f1_weighted, 4),
                "confusion_matrix": cm,
                "classification_report": report
            }

        xgb_eval = calculate_metrics(y_test, xgb_preds)
        rf_eval = calculate_metrics(y_test, rf_preds)

        # Save metadata
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_metadata = {
            "model_id": model_id,
            "trained_at": datetime.now().isoformat(),
            "classes": classes,
            "feature_columns": self.feature_columns,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "split_method": split_method,
            "xgboost_eval": xgb_eval,
            "random_forest_eval": rf_eval,
            "hyperparameters": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate
            }
        }

        # Save to disk
        self.save_model(model_id)

        return self.model_metadata

    def predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Runs inference using trained XGBoost model."""
        if self.xgb_model is None:
            raise RuntimeError("No model is trained or loaded. Please train or load a model first.")

        # Ensure features present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_columns].fillna(0).values
        preds = self.xgb_model.predict(X)
        probs = self.xgb_model.predict_proba(X)
        classes = list(self.label_encoder.classes_)

        results = []
        for i, (pred_idx, prob_dist) in enumerate(zip(preds, probs)):
            predicted_class = classes[pred_idx]
            confidence = float(np.max(prob_dist))
            is_anomaly = predicted_class != "NORMAL"

            # Derive severity
            if not is_anomaly:
                severity = "NORMAL"
            elif confidence > 0.85 or predicted_class in ["SERVICE_FAILURE", "DATABASE_FAILURE"]:
                severity = "CRITICAL"
            else:
                severity = "WARNING"

            prob_dict = {classes[j]: round(float(prob_dist[j]), 4) for j in range(len(classes))}

            results.append({
                "index": i,
                "service": str(df.iloc[i].get("service", "unknown")),
                "window": str(df.iloc[i].get("window", "")),
                "is_anomaly": is_anomaly,
                "predicted_category": predicted_class,
                "confidence": round(confidence, 4),
                "severity": severity,
                "probabilities": prob_dict,
                "features": {col: round(float(df.iloc[i][col]), 2) for col in self.feature_columns}
            })

        return results

    def save_model(self, model_id: str):
        save_path = MODELS_DIR / f"{model_id}.joblib"
        joblib.dump({
            "xgb_model": self.xgb_model,
            "rf_model": self.rf_model,
            "label_encoder": self.label_encoder,
            "metadata": self.model_metadata
        }, save_path)
        # Also save latest pointer
        latest_path = MODELS_DIR / "latest_model.joblib"
        joblib.dump(save_path, latest_path)

    def load_model(self, model_id: Optional[str] = None):
        if model_id:
            load_path = MODELS_DIR / f"{model_id}.joblib"
        else:
            latest_path = MODELS_DIR / "latest_model.joblib"
            if not latest_path.exists():
                return False
            load_path = joblib.load(latest_path)

        if not os.path.exists(load_path):
            return False

        data = joblib.load(load_path)
        self.xgb_model = data["xgb_model"]
        self.rf_model = data["rf_model"]
        self.label_encoder = data["label_encoder"]
        self.model_metadata = data["metadata"]
        self.feature_columns = self.model_metadata.get("feature_columns", self.feature_columns)
        return True

    @staticmethod
    def list_saved_models() -> List[Dict[str, Any]]:
        models = []
        for f in MODELS_DIR.glob("model_*.joblib"):
            try:
                data = joblib.load(f)
                meta = data.get("metadata", {})
                models.append({
                    "model_id": meta.get("model_id", f.stem),
                    "trained_at": meta.get("trained_at"),
                    "classes": meta.get("classes", []),
                    "xgb_accuracy": meta.get("xgboost_eval", {}).get("accuracy"),
                    "xgb_f1": meta.get("xgboost_eval", {}).get("f1_score"),
                    "rf_accuracy": meta.get("random_forest_eval", {}).get("accuracy"),
                    "rf_f1": meta.get("random_forest_eval", {}).get("f1_score"),
                    "train_samples": meta.get("train_samples")
                })
            except Exception:
                pass
        return sorted(models, key=lambda x: x.get("trained_at", ""), reverse=True)
