import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from app.config import ANOMALY_CATEGORIES

class TelemetryLabelingEngine:
    """Manages rule-assisted labeling, manual labeling, and unsupervised anomaly detection."""

    # Default rule thresholds for rule-assisted labeling
    DEFAULT_RULES = {
        "high_cpu_threshold": 80.0,       # CPU utilization %
        "high_memory_threshold": 85.0,    # Memory utilization %
        "high_latency_threshold": 400.0,  # P95 latency ms
        "high_error_rate_threshold": 0.15,# 15% error rate
        "high_db_duration_threshold": 250.0,# DB query duration ms
        "restart_threshold": 1            # Pod restart count
    }

    @staticmethod
    def apply_rule_assisted_labels(
        df: pd.DataFrame,
        custom_rules: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Applies heuristic rule-assisted labels with symptom vs probable cause distinction."""
        rules = TelemetryLabelingEngine.DEFAULT_RULES.copy()
        if custom_rules:
            rules.update(custom_rules)

        labeled_df = df.copy()
        assigned_labels = []
        rule_reasons = []

        for _, row in labeled_df.iterrows():
            cpu = row.get("cpu_util", 20.0)
            mem = row.get("memory_util", 40.0)
            p95_lat = row.get("p95_latency", 50.0)
            err_rate = row.get("error_rate", 0.0)
            db_dur = row.get("db_duration_avg", 10.0)
            restarts = row.get("restart_count", 0)
            failed_spans = row.get("failed_span_count", 0)

            label = "NORMAL"
            reasons = []

            # Prioritized rules: Root cause > Symptoms
            if restarts >= rules["restart_threshold"] or mem >= 92.0:
                label = "SERVICE_FAILURE"
                reasons.append(f"Pod restarts ({restarts}) or critical memory pressure ({mem:.1f}%)")
            elif db_dur >= rules["high_db_duration_threshold"]:
                label = "DATABASE_FAILURE"
                reasons.append(f"Database query latency ({db_dur:.1f}ms) exceeded threshold ({rules['high_db_duration_threshold']}ms)")
            elif cpu >= rules["high_cpu_threshold"]:
                label = "HIGH_CPU"
                reasons.append(f"CPU utilization ({cpu:.1f}%) exceeded threshold ({rules['high_cpu_threshold']}%)")
            elif mem >= rules["high_memory_threshold"]:
                label = "HIGH_MEMORY"
                reasons.append(f"Memory utilization ({mem:.1f}%) exceeded threshold ({rules['high_memory_threshold']}%)")
            elif failed_spans > 5 and err_rate > 0.3:
                label = "NETWORK_FAILURE"
                reasons.append(f"High failed spans ({failed_spans}) and high error rate ({err_rate:.2f})")
            elif err_rate >= rules["high_error_rate_threshold"]:
                label = "HIGH_ERROR_RATE"
                reasons.append(f"Error rate ({err_rate * 100:.1f}%) exceeded threshold ({rules['high_error_rate_threshold'] * 100}%)")
            elif p95_lat >= rules["high_latency_threshold"]:
                label = "HIGH_LATENCY"
                reasons.append(f"P95 latency ({p95_lat:.1f}ms) exceeded threshold ({rules['high_latency_threshold']}ms)")
            else:
                label = "NORMAL"
                reasons.append("All metrics within operational baselines")

            assigned_labels.append(label)
            rule_reasons.append("; ".join(reasons))

        labeled_df["suggested_label"] = assigned_labels
        labeled_df["rule_reason"] = rule_reasons
        if "label" not in labeled_df.columns:
            labeled_df["label"] = assigned_labels

        return labeled_df

    @staticmethod
    def run_unsupervised_anomaly_detection(
        df: pd.DataFrame,
        feature_columns: List[str],
        contamination: float = 0.05
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Provides unsupervised anomaly detection via Isolation Forest for unlabeled datasets."""
        X = df[feature_columns].fillna(0).values
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        preds = iso_forest.fit_predict(X) # 1 = normal, -1 = anomaly
        scores = -iso_forest.score_samples(X) # Higher score = more anomalous

        result_df = df.copy()
        result_df["unsupervised_is_anomaly"] = [bool(p == -1) for p in preds]
        result_df["unsupervised_anomaly_score"] = scores.round(4)
        
        anomaly_count = int((preds == -1).sum())
        total_count = len(preds)

        summary = {
            "method": "Isolation Forest",
            "total_records": total_count,
            "detected_anomalies": anomaly_count,
            "anomaly_percentage": round(float(anomaly_count / max(total_count, 1) * 100), 2),
            "contamination_param": contamination,
            "notice": "Unsupervised anomaly detection identifies statistical outliers without human ground-truth labels. For supervised root-cause classification, labels must be assigned or verified."
        }
        return result_df, summary
