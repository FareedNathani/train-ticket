import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add backend directory to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.config import TRAIN_TICKET_SERVICES
from app.parsers import TelemetryParser
from app.pipeline import TelemetryFeaturePipeline
from app.labeling import TelemetryLabelingEngine
from app.ml_engine import MLEngine
from app.xai_engine import XAIEngine
from app.rca_engine import RootCauseAnalysisEngine
from app.synthetic_generator import TrainTicketBenchmarkGenerator

def test_telemetry_parser():
    # Test Java Spring Boot log parsing
    raw_log = "2026-11-08 14:22:15.123 [http-nio-8080-exec-1] ERROR ts-order-service [traceId=trc-001,spanId=spn-002] c.f.t.OrderController - com.mongodb.MongoTimeoutException: Connection pool exhausted"
    parsed = TelemetryParser.parse_log_line(raw_log)
    assert parsed is not None
    assert parsed["level"] == "ERROR"
    assert parsed["service"] == "ts-order-service"
    assert "MongoTimeoutException" in parsed["message"]
    assert parsed["trace_id"] == "trc-001"

    # Test JSON log parsing
    json_log = '{"timestamp": "2026-11-08T14:22:15", "service": "ts-travel-service", "level": "WARN", "message": "Circuit breaker slow call rate exceeded"}'
    parsed_json = TelemetryParser.parse_log_line(json_log)
    assert parsed_json["level"] == "WARN"
    assert parsed_json["service"] == "ts-travel-service"

def test_data_quality_inspector():
    df = pd.DataFrame([
        {"timestamp": "2026-11-08T10:00:00", "service": "ts-order-service", "cpu": 55.0, "latency": 120.0},
        {"timestamp": "2026-11-08T10:00:30", "service": "ts-order-service", "cpu": None, "latency": 150.0},
        {"timestamp": "2026-11-08T10:00:00", "service": "ts-order-service", "cpu": 55.0, "latency": 120.0} # Duplicate
    ])
    profile = TelemetryParser.inspect_data_quality(df, "test_metrics.csv")
    assert profile["total_rows"] == 3
    assert profile["duplicates"] == 1
    assert profile["columns"][2]["missing"] == 1 # cpu column missing 1 value

def test_synthetic_benchmark_generator():
    paths = TrainTicketBenchmarkGenerator.generate_benchmark_datasets(num_windows=30)
    assert "logs" in paths
    assert "metrics" in paths
    assert "traces" in paths
    assert "events" in paths
    assert "benchmark" in paths

    df = pd.read_csv(paths["benchmark"])
    assert len(df) > 50
    assert "p95_latency" in df.columns
    assert "label" in df.columns
    assert "DATABASE_FAILURE" in df["label"].values

def test_ml_and_xai_pipeline():
    # Generate small benchmark
    paths = TrainTicketBenchmarkGenerator.generate_benchmark_datasets(num_windows=40)
    df = pd.read_csv(paths["benchmark"])

    ml = MLEngine()
    report = ml.train_models(df, label_col="label", test_size=0.25, n_estimators=30, max_depth=4)

    # Check metrics existence
    assert "xgboost_eval" in report
    assert "random_forest_eval" in report
    assert report["xgboost_eval"]["accuracy"] >= 0.70
    assert "confusion_matrix" in report["xgboost_eval"]

    # Test inference
    predictions = ml.predict(df.head(5))
    assert len(predictions) == 5
    assert "predicted_category" in predictions[0]
    assert "confidence" in predictions[0]

    # Test SHAP XAI
    xai = XAIEngine(ml)
    global_xai = xai.explain_global(df.head(20))
    assert len(global_xai["global_importances"]) > 0

    instance_xai = xai.explain_instance(df.iloc[0])
    assert "natural_language_explanation" in instance_xai
    assert "contributions" in instance_xai
    assert len(instance_xai["contributions"]) > 0

def test_rca_engine():
    rca = RootCauseAnalysisEngine()
    topo = rca.get_topology_data()
    assert len(topo["nodes"]) >= len(TRAIN_TICKET_SERVICES)

    # Check blast radius of mongo-db
    blast = rca.find_blast_radius("mongo-db")
    assert "ts-order-service" in blast["upstream_callers_affected"]

    # Test 3-tier diagnosis
    logs = [{
        "timestamp": "2026-11-08T12:00:00",
        "service": "ts-order-service",
        "level": "ERROR",
        "message": "com.mongodb.MongoTimeoutException: connection pool exhausted",
        "exception": "com.mongodb.MongoTimeoutException"
    }]
    shap_info = {
        "top_increasing_features": [{"feature": "db_duration_avg", "value": 650.0}],
        "natural_language_explanation": "Database latency spiked."
    }
    diag = rca.diagnose_anomaly(
        predicted_category="DATABASE_FAILURE",
        affected_service="ts-order-service",
        shap_explanation=shap_info,
        raw_logs=logs
    )
    # With explicit log error, tier must be CONFIRMED_ROOT_CAUSE
    assert diag["diagnosis_tier"] == "CONFIRMED_ROOT_CAUSE"
    assert "mongo-db" in diag["root_cause_service"]
    assert len(diag["recommended_troubleshooting_steps"]) > 0

if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
