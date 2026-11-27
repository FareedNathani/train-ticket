import os
import re
import json
import csv
from io import StringIO
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Regular expressions for multi-language log parsing
# Java Spring Boot: 2026-11-08 14:22:15.123 [http-nio-8080-exec-1] INFO  ts-order-service [traceId=abc123,spanId=def456] c.f.t.OrderController - Order created successfully
JAVA_SPRING_LOG_REGEX = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\s+'
    r'(?:\[(?P<thread>[^\]]*)\]\s+)?'
    r'(?P<level>[A-Z]{4,5})\s+'
    r'(?:\[?(?P<service>[a-zA-Z0-9_\-\.]+)(?:,[a-zA-Z0-9_\-]+)?\]?\s+)?'
    r'(?:\[(?:traceId=)?(?P<trace_id>[a-zA-Z0-9\-]+)?(?:,\s*(?:spanId=)?(?P<span_id>[a-zA-Z0-9\-]+))?\]\s+)?'
    r'(?:(?P<logger>[a-zA-Z0-9_\.\$]+)\s*[:-]\s*)?'
    r'(?P<message>.*)$'
)

# Standard Syslog / Generic Log Regex
GENERIC_LOG_REGEX = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+'
    r'(?:\[(?P<level>[A-Z]+)\]|(?P<level_alt>[A-Z]{4,5}))\s+'
    r'(?:\[(?P<service>[a-zA-Z0-9_\-\.]+)\]\s+)?'
    r'(?P<message>.*)$'
)

class TelemetryParser:
    """Parses logs, metrics, traces, and events from diverse formats."""

    @staticmethod
    def parse_log_line(line: str, default_service: str = "unknown-service") -> Optional[Dict[str, Any]]:
        line = line.strip()
        if not line:
            return None

        # 1. Try parsing as JSON first
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                return {
                    "timestamp": data.get("timestamp") or data.get("time") or data.get("@timestamp") or datetime.now().isoformat(),
                    "service": data.get("service") or data.get("service_name") or data.get("app") or default_service,
                    "level": (data.get("level") or data.get("severity") or "INFO").upper(),
                    "message": data.get("message") or data.get("msg") or "",
                    "trace_id": data.get("trace_id") or data.get("traceId") or "",
                    "span_id": data.get("span_id") or data.get("spanId") or "",
                    "exception": data.get("exception") or data.get("stack_trace") or data.get("error") or ""
                }
            except Exception:
                pass

        # 2. Try Java Spring Boot Regex
        match = JAVA_SPRING_LOG_REGEX.match(line)
        if match:
            group_dict = match.groupdict()
            service = group_dict.get("service") or default_service
            # Clean service name if it was part of logger
            return {
                "timestamp": group_dict.get("timestamp"),
                "service": service,
                "level": (group_dict.get("level") or "INFO").upper(),
                "message": group_dict.get("message") or "",
                "trace_id": group_dict.get("trace_id") or "",
                "span_id": group_dict.get("span_id") or "",
                "exception": ""
            }

        # 3. Try Generic Log Regex
        match2 = GENERIC_LOG_REGEX.match(line)
        if match2:
            gd = match2.groupdict()
            lvl = gd.get("level") or gd.get("level_alt") or "INFO"
            return {
                "timestamp": gd.get("timestamp"),
                "service": gd.get("service") or default_service,
                "level": lvl.upper(),
                "message": gd.get("message") or "",
                "trace_id": "",
                "span_id": "",
                "exception": ""
            }

        # Fallback to plain line
        return {
            "timestamp": datetime.now().isoformat(),
            "service": default_service,
            "level": "INFO" if "ERROR" not in line.upper() and "WARN" not in line.upper() else ("ERROR" if "ERROR" in line.upper() else "WARN"),
            "message": line,
            "trace_id": "",
            "span_id": "",
            "exception": ""
        }

    @staticmethod
    def parse_raw_text(content: str, default_service: str = "ts-service") -> List[Dict[str, Any]]:
        records = []
        current_record = None

        for line in content.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            parsed = TelemetryParser.parse_log_line(line_str, default_service=default_service)
            if parsed and parsed.get("timestamp"):
                if current_record:
                    records.append(current_record)
                current_record = parsed
            else:
                # Appending stacktrace or multiline continuation
                if current_record:
                    current_record["message"] += "\n" + line_str
                    if "exception" in line_str.lower() or "error" in line_str.lower() or "at " in line_str:
                        current_record["exception"] = (current_record["exception"] + "\n" + line_str).strip()
                else:
                    current_record = {
                        "timestamp": datetime.now().isoformat(),
                        "service": default_service,
                        "level": "INFO",
                        "message": line_str,
                        "trace_id": "",
                        "span_id": "",
                        "exception": ""
                    }

        if current_record:
            records.append(current_record)

        return records

    @staticmethod
    def load_file_content(content_str: str, filename: str) -> pd.DataFrame:
        """Parses CSV, JSON, JSONL, or TXT into a standard pandas DataFrame."""
        fn = filename.lower()
        if fn.endswith(".csv"):
            return pd.read_csv(StringIO(content_str))
        elif fn.endswith(".jsonl"):
            lines = [json.loads(l) for l in content_str.splitlines() if l.strip()]
            return pd.DataFrame(lines)
        elif fn.endswith(".json"):
            data = json.loads(content_str)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check for standard wrapped keys
                for key in ["records", "data", "items", "telemetry", "spans", "logs", "metrics", "events"]:
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])
                return pd.DataFrame([data])
        else: # TXT or log file
            parsed = TelemetryParser.parse_raw_text(content_str)
            return pd.DataFrame(parsed)

    @staticmethod
    def detect_telemetry_type(df: pd.DataFrame) -> str:
        """Detects whether dataset contains logs, metrics, traces, or events."""
        cols = {c.lower() for c in df.columns}

        # Traces
        if {"trace_id", "span_id"}.issubset(cols) or {"traceid", "spanid"}.issubset(cols) or "parent_span_id" in cols or "duration" in cols:
            return "traces"

        # Events
        if {"event_type", "reason"}.intersection(cols) or "event" in cols or "restart" in cols or "oomkilled" in str(df.values).lower():
            return "events"

        # Metrics
        metric_keywords = {"cpu", "cpu_util", "memory", "mem_util", "latency", "p95", "throughput", "rps", "request_rate", "disk", "network"}
        if len(cols.intersection(metric_keywords)) >= 2:
            return "metrics"

        # Logs
        log_keywords = {"level", "message", "msg", "log", "exception", "stack_trace", "logger"}
        if len(cols.intersection(log_keywords)) >= 1:
            return "logs"

        # Mixed / Default
        return "telemetry"

    @staticmethod
    def inspect_data_quality(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Provides full quality profiling: missing values, duplicates, types, time span."""
        total_rows = len(df)
        total_cols = len(df.columns)
        detected_type = TelemetryParser.detect_telemetry_type(df)

        missing_counts = df.isnull().sum().to_dict()
        missing_pcts = {col: round(float(count / max(total_rows, 1) * 100), 2) for col, count in missing_counts.items()}
        duplicates = int(df.duplicated().sum())

        columns_info = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            sample_val = None
            non_null = df[col].dropna()
            if not non_null.empty:
                sample_val = str(non_null.iloc[0])[:100]
            columns_info.append({
                "name": col,
                "type": dtype_str,
                "missing": int(missing_counts[col]),
                "missing_pct": missing_pcts[col],
                "sample": sample_val
            })

        # Identify timestamp column
        ts_col = None
        time_range = {"start": None, "end": None}
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                ts_col = col
                try:
                    ts_series = pd.to_datetime(df[col], errors="coerce").dropna()
                    if not ts_series.empty:
                        time_range["start"] = str(ts_series.min())
                        time_range["end"] = str(ts_series.max())
                except Exception:
                    pass
                break

        # Identify service names
        service_col = None
        services_found = []
        for col in df.columns:
            if col.lower() in ["service", "service_name", "app", "microservice", "component"]:
                service_col = col
                services_found = sorted([str(s) for s in df[col].dropna().unique().tolist() if s])
                break

        preview_records = df.head(10).replace({np.nan: None}).to_dict(orient="records")

        return {
            "filename": filename,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "detected_type": detected_type,
            "duplicates": duplicates,
            "timestamp_column": ts_col,
            "time_range": time_range,
            "service_column": service_col,
            "services_found": services_found[:20],
            "total_unique_services": len(services_found),
            "columns": columns_info,
            "preview": preview_records
        }

    @staticmethod
    def standardize_fields(df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Standardizes column names based on mapping or auto-detection."""
        df_clean = df.copy()
        if mapping:
            df_clean = df_clean.rename(columns=mapping)

        # Canonical auto-rename dictionary
        auto_map = {
            "time": "timestamp",
            "@timestamp": "timestamp",
            "datetime": "timestamp",
            "service_name": "service",
            "microservice": "service",
            "app": "service",
            "log_level": "level",
            "severity": "level",
            "msg": "message",
            "err": "exception",
            "stacktrace": "exception",
            "traceid": "trace_id",
            "spanid": "span_id",
            "parentspanid": "parent_span_id",
            "duration_ms": "duration",
            "latency_ms": "latency"
        }
        for orig, canon in auto_map.items():
            if orig in df_clean.columns and canon not in df_clean.columns:
                df_clean = df_clean.rename(columns={orig: canon})

        # Ensure timestamp is normalized
        if "timestamp" in df_clean.columns:
            df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")
            df_clean = df_clean.dropna(subset=["timestamp"])

        return df_clean
