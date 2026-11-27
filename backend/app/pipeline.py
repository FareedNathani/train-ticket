import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from app.config import PROCESSED_DIR, MODELS_DIR

# Key exception signatures in microservices
EXCEPTION_PATTERNS = {
    "db_connection_error": ["mongotimeoutexception", "com.mongodb", "sqlexception", "connection refused", "communications link failure"],
    "timeout_error": ["sockettimeoutexception", "read timeout", "connection timed out", "gateway timeout", "upstream request timeout"],
    "oom_error": ["outofmemoryerror", "oomkilled", "java heap space", "memory allocation failed"],
    "service_unavailable": ["503 service unavailable", "circuit breaker open", "connection reset by peer"],
    "auth_error": ["unauthorized", "invalid token", "authentication failed", "forbidden 403"]
}

class TelemetryFeaturePipeline:
    """Preprocesses and aggregates telemetry into unified time-window feature sets."""

    FEATURE_COLUMNS = [
        "error_count",
        "warning_count",
        "error_rate",
        "avg_latency",
        "p95_latency",
        "p99_latency",
        "cpu_util",
        "memory_util",
        "request_throughput",
        "failed_span_count",
        "trace_duration_avg",
        "db_duration_avg",
        "restart_count",
        "downstream_delay_ratio",
        "affected_service_count"
    ]

    def __init__(self, window_seconds: int = 30):
        self.window_seconds = window_seconds
        self.scaler = StandardScaler()
        self.is_fitted = False

    @staticmethod
    def standardize_service_name(service: str) -> str:
        if not service or not isinstance(service, str):
            return "unknown-service"
        s = service.strip().lower()
        # Clean common prefixes/suffixes
        s = s.replace("service/", "").replace("app:", "")
        return s

    @staticmethod
    def detect_exception_types(text: str) -> Dict[str, int]:
        results = {k: 0 for k in EXCEPTION_PATTERNS}
        if not text or not isinstance(text, str):
            return results
        text_lower = text.lower()
        for cat, keywords in EXCEPTION_PATTERNS.items():
            if any(k in text_lower for k in keywords):
                results[cat] = 1
        return results

    def aggregate_multi_source_telemetry(
        self,
        logs_df: Optional[pd.DataFrame] = None,
        metrics_df: Optional[pd.DataFrame] = None,
        traces_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        window_size: str = "30s"
    ) -> pd.DataFrame:
        """Aggregates logs, metrics, traces, and events into consistent time-window feature vectors."""

        records_by_key = {}  # (window_timestamp, service) -> aggregated dict

        def get_bucket(ts):
            if pd.isna(ts):
                return None
            ts_dt = pd.to_datetime(ts)
            # Round down to window
            return ts_dt.floor(window_size)

        # 1. Process Metrics if present
        if metrics_df is not None and not metrics_df.empty:
            m_df = metrics_df.copy()
            if "timestamp" in m_df.columns and "service" in m_df.columns:
                m_df["window"] = m_df["timestamp"].apply(get_bucket)
                m_df["service"] = m_df["service"].apply(self.standardize_service_name)
                
                # Group by window and service
                for (win, srv), group in m_df.groupby(["window", "service"]):
                    if pd.isna(win):
                        continue
                    key = (win, srv)
                    if key not in records_by_key:
                        records_by_key[key] = {"window": win, "service": srv}

                    records_by_key[key]["cpu_util"] = float(group["cpu_util"].mean()) if "cpu_util" in group.columns else float(group.get("cpu", pd.Series([20.0])).mean())
                    records_by_key[key]["memory_util"] = float(group["memory_util"].mean()) if "memory_util" in group.columns else float(group.get("memory", pd.Series([40.0])).mean())
                    
                    if "latency" in group.columns:
                        lat = group["latency"].dropna()
                        if not lat.empty:
                            records_by_key[key]["avg_latency"] = float(lat.mean())
                            records_by_key[key]["p95_latency"] = float(lat.quantile(0.95))
                            records_by_key[key]["p99_latency"] = float(lat.quantile(0.99))
                    
                    if "throughput" in group.columns or "rps" in group.columns:
                        tp_col = "throughput" if "throughput" in group.columns else "rps"
                        records_by_key[key]["request_throughput"] = float(group[tp_col].sum())

        # 2. Process Logs if present
        if logs_df is not None and not logs_df.empty:
            l_df = logs_df.copy()
            if "timestamp" in l_df.columns:
                l_df["window"] = l_df["timestamp"].apply(get_bucket)
                if "service" not in l_df.columns:
                    l_df["service"] = "unknown-service"
                l_df["service"] = l_df["service"].apply(self.standardize_service_name)
                
                for (win, srv), group in l_df.groupby(["window", "service"]):
                    if pd.isna(win):
                        continue
                    key = (win, srv)
                    if key not in records_by_key:
                        records_by_key[key] = {"window": win, "service": srv}

                    levels = group["level"].str.upper().value_counts().to_dict() if "level" in group.columns else {}
                    err_cnt = levels.get("ERROR", 0) + levels.get("FATAL", 0)
                    warn_cnt = levels.get("WARN", 0) + levels.get("WARNING", 0)
                    total_logs = len(group)

                    records_by_key[key]["error_count"] = records_by_key[key].get("error_count", 0) + err_cnt
                    records_by_key[key]["warning_count"] = records_by_key[key].get("warning_count", 0) + warn_cnt
                    records_by_key[key]["error_rate"] = float(err_cnt / max(total_logs, 1))

        # 3. Process Traces if present
        if traces_df is not None and not traces_df.empty:
            t_df = traces_df.copy()
            if "timestamp" in t_df.columns or "start_time" in t_df.columns:
                ts_col = "timestamp" if "timestamp" in t_df.columns else "start_time"
                t_df["window"] = t_df[ts_col].apply(get_bucket)
                if "service" not in t_df.columns:
                    t_df["service"] = "unknown-service"
                t_df["service"] = t_df["service"].apply(self.standardize_service_name)

                for (win, srv), group in t_df.groupby(["window", "service"]):
                    if pd.isna(win):
                        continue
                    key = (win, srv)
                    if key not in records_by_key:
                        records_by_key[key] = {"window": win, "service": srv}

                    dur_col = "duration" if "duration" in group.columns else "duration_ms"
                    if dur_col in group.columns:
                        durs = group[dur_col].dropna()
                        if not durs.empty:
                            records_by_key[key]["trace_duration_avg"] = float(durs.mean())
                            if "avg_latency" not in records_by_key[key]:
                                records_by_key[key]["avg_latency"] = float(durs.mean())
                                records_by_key[key]["p95_latency"] = float(durs.quantile(0.95))
                                records_by_key[key]["p99_latency"] = float(durs.quantile(0.99))

                    # Failed spans
                    failed_count = 0
                    if "has_error" in group.columns:
                        failed_count = int(group["has_error"].sum())
                    elif "status" in group.columns or "status_code" in group.columns:
                        stat_col = "status" if "status" in group.columns else "status_code"
                        failed_count = int(group[stat_col].astype(str).str.contains("error|50|fail|40", case=False).sum())
                    records_by_key[key]["failed_span_count"] = failed_count

                    # Database duration if operation includes DB
                    if "operation_name" in group.columns or "operation" in group.columns:
                        op_col = "operation_name" if "operation_name" in group.columns else "operation"
                        db_spans = group[group[op_col].astype(str).str.contains("mongo|sql|db|query|find|insert|update", case=False)]
                        if not db_spans.empty and dur_col in db_spans.columns:
                            records_by_key[key]["db_duration_avg"] = float(db_spans[dur_col].mean())

        # 4. Process Operational Events if present
        if events_df is not None and not events_df.empty:
            e_df = events_df.copy()
            if "timestamp" in e_df.columns:
                e_df["window"] = e_df["timestamp"].apply(get_bucket)
                if "service" not in e_df.columns:
                    e_df["service"] = "unknown-service"
                e_df["service"] = e_df["service"].apply(self.standardize_service_name)

                for (win, srv), group in e_df.groupby(["window", "service"]):
                    if pd.isna(win):
                        continue
                    key = (win, srv)
                    if key not in records_by_key:
                        records_by_key[key] = {"window": win, "service": srv}

                    restarts = 0
                    for _, row in group.iterrows():
                        ev_str = f"{row.get('event_type', '')} {row.get('reason', '')} {row.get('message', '')}".lower()
                        if "restart" in ev_str or "oomkilled" in ev_str or "crashloopbackoff" in ev_str or "failed" in ev_str:
                            restarts += 1
                    records_by_key[key]["restart_count"] = restarts

        # Convert to DataFrame
        if not records_by_key:
            return pd.DataFrame(columns=["window", "service"] + self.FEATURE_COLUMNS)

        feature_df = pd.DataFrame(list(records_by_key.values()))

        # Fill missing features with reasonable defaults
        defaults = {
            "error_count": 0,
            "warning_count": 0,
            "error_rate": 0.0,
            "avg_latency": 45.0, # normal ms
            "p95_latency": 65.0,
            "p99_latency": 90.0,
            "cpu_util": 25.0,    # normal %
            "memory_util": 45.0, # normal %
            "request_throughput": 80.0,
            "failed_span_count": 0,
            "trace_duration_avg": 50.0,
            "db_duration_avg": 12.0,
            "restart_count": 0,
            "downstream_delay_ratio": 0.1,
            "affected_service_count": 1
        }
        for col, val in defaults.items():
            if col not in feature_df.columns:
                feature_df[col] = val
            else:
                feature_df[col] = feature_df[col].fillna(val)

        # Calculate downstream delay ratio
        feature_df["downstream_delay_ratio"] = (
            (feature_df["avg_latency"] - feature_df["db_duration_avg"]).clip(lower=0) /
            (feature_df["avg_latency"] + 1e-5)
        ).clip(0.0, 1.0)

        # Count affected services in the same time window
        window_counts = feature_df.groupby("window")["service"].transform("nunique")
        feature_df["affected_service_count"] = window_counts

        # Sort chronologically
        feature_df = feature_df.sort_values(by=["window", "service"]).reset_index(drop=True)
        return feature_df

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Scales numerical features for training and prevents leakage."""
        X = df[self.FEATURE_COLUMNS].copy().values
        scaled_X = self.scaler.fit_transform(X)
        self.is_fitted = True
        return scaled_X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Applies fitted scaler during inference."""
        if not self.is_fitted:
            # Fit on incoming if not yet fitted
            return self.fit_transform(df)
        X = df[self.FEATURE_COLUMNS].copy().values
        return self.scaler.transform(X)

    def save(self, filepath: str):
        joblib.dump({"scaler": self.scaler, "is_fitted": self.is_fitted, "columns": self.FEATURE_COLUMNS}, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.is_fitted = data["is_fitted"]
