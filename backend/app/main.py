import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.config import (
    UPLOAD_DIR, PROCESSED_DIR, SYNTHETIC_DIR, MODELS_DIR,
    FRONTEND_DIR, TRAIN_TICKET_SERVICES
)
from app.parsers import TelemetryParser
from app.pipeline import TelemetryFeaturePipeline
from app.labeling import TelemetryLabelingEngine
from app.ml_engine import MLEngine
from app.xai_engine import XAIEngine
from app.rca_engine import RootCauseAnalysisEngine
from app.synthetic_generator import TrainTicketBenchmarkGenerator

# Initialize FastAPI application
app = FastAPI(
    title="Train Ticket AI Observability & Root Cause Analysis Engine",
    description="Intelligent anomaly detection, SHAP explainability, and multi-source root cause diagnosis for microservices.",
    version="1.0.0"
)

# Enable CORS for dashboard and API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core Engines State
pipeline = TelemetryFeaturePipeline()
ml_engine = MLEngine()
xai_engine = XAIEngine(ml_engine)
rca_engine = RootCauseAnalysisEngine()

# In-memory working cache for active datasets
active_state = {
    "current_dataset": None,
    "current_dataset_path": None,
    "current_df": None,
    "current_profile": None,
    "last_prediction": None,
    "raw_logs": [],
    "raw_traces": [],
    "raw_events": []
}

# Auto-load latest model if present
ml_engine.load_model()

# Auto-generate benchmark on startup if not present
benchmark_file = SYNTHETIC_DIR / "train_ticket_aggregated_benchmark.csv"
if not benchmark_file.exists():
    TrainTicketBenchmarkGenerator.generate_benchmark_datasets(num_windows=120)

if benchmark_file.exists():
    try:
        active_state["current_df"] = pd.read_csv(benchmark_file)
        active_state["current_dataset"] = "train_ticket_aggregated_benchmark.csv"
        active_state["current_dataset_path"] = str(benchmark_file)
        active_state["current_profile"] = TelemetryParser.inspect_data_quality(
            active_state["current_df"], "train_ticket_aggregated_benchmark.csv"
        )
    except Exception:
        pass


# -------------------------------------------------------------
# Request & Response Models
# -------------------------------------------------------------
class RuleLabelRequest(BaseModel):
    high_cpu_threshold: Optional[float] = 80.0
    high_memory_threshold: Optional[float] = 85.0
    high_latency_threshold: Optional[float] = 400.0
    high_error_rate_threshold: Optional[float] = 0.15
    high_db_duration_threshold: Optional[float] = 250.0
    restart_threshold: Optional[int] = 1

class TrainRequest(BaseModel):
    test_size: Optional[float] = 0.25
    split_method: Optional[str] = "stratified"
    n_estimators: Optional[int] = 120
    max_depth: Optional[int] = 6
    learning_rate: Optional[float] = 0.1

class InferenceRequest(BaseModel):
    records: Optional[List[Dict[str, Any]]] = None

class ManualLabelRequest(BaseModel):
    index: int
    label: str

class DiagnoseRequest(BaseModel):
    record_index: Optional[int] = 0
    service: Optional[str] = "ts-order-service"
    predicted_category: Optional[str] = "DATABASE_FAILURE"


# -------------------------------------------------------------
# System & Topology Endpoints
# -------------------------------------------------------------
@app.get("/api/health")
def get_system_health():
    return {
        "status": "ONLINE",
        "loaded_model": ml_engine.model_metadata.get("model_id"),
        "model_trained_at": ml_engine.model_metadata.get("trained_at"),
        "model_classes": list(ml_engine.label_encoder.classes_) if hasattr(ml_engine.label_encoder, "classes_") else [],
        "active_dataset": active_state["current_dataset"],
        "active_records": len(active_state["current_df"]) if active_state["current_df"] is not None else 0,
        "supported_services_count": len(TRAIN_TICKET_SERVICES)
    }

@app.get("/api/topology")
def get_topology():
    return rca_engine.get_topology_data()


# -------------------------------------------------------------
# Telemetry Ingestion & Profiling Endpoints
# -------------------------------------------------------------
@app.post("/api/datasets/synthetic/generate")
def generate_synthetic_benchmark(windows: int = 120):
    paths = TrainTicketBenchmarkGenerator.generate_benchmark_datasets(num_windows=windows)
    
    # Load aggregated benchmark into active memory
    benchmark_file = Path(paths["benchmark"])
    active_state["current_df"] = pd.read_csv(benchmark_file)
    active_state["current_dataset"] = benchmark_file.name
    active_state["current_dataset_path"] = str(benchmark_file)
    active_state["current_profile"] = TelemetryParser.inspect_data_quality(
        active_state["current_df"], benchmark_file.name
    )

    # Load raw telemetry caches for RCA
    try:
        active_state["raw_logs"] = pd.read_csv(paths["logs"]).to_dict(orient="records")
        with open(paths["traces"], "r") as f:
            active_state["raw_traces"] = [json.loads(line) for line in f if line.strip()]
        with open(paths["events"], "r") as f:
            active_state["raw_events"] = json.load(f)
    except Exception:
        pass

    return {
        "message": f"Successfully generated Train Ticket benchmark dataset with {windows} synchronized operational windows.",
        "files": paths,
        "profile": active_state["current_profile"]
    }

@app.post("/api/telemetry/upload")
async def upload_telemetry_file(file: UploadFile = File(...)):
    filename = file.filename
    content_bytes = await file.read()
    content_str = content_bytes.decode("utf-8", errors="replace")

    # Save raw upload
    dest_path = UPLOAD_DIR / filename
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(content_str)

    try:
        df = TelemetryParser.load_file_content(content_str, filename)
        df_std = TelemetryParser.standardize_fields(df)

        profile = TelemetryParser.inspect_data_quality(df_std, filename)

        active_state["current_df"] = df_std
        active_state["current_dataset"] = filename
        active_state["current_dataset_path"] = str(dest_path)
        active_state["current_profile"] = profile

        # If logs/traces/events, cache in memory for RCA correlation
        if profile["detected_type"] == "logs":
            active_state["raw_logs"] = df_std.to_dict(orient="records")
        elif profile["detected_type"] == "traces":
            active_state["raw_traces"] = df_std.to_dict(orient="records")
        elif profile["detected_type"] == "events":
            active_state["raw_events"] = df_std.to_dict(orient="records")

        return {
            "message": f"Successfully parsed and profiled '{filename}'",
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse telemetry file: {str(e)}")

@app.get("/api/telemetry/profile")
def get_current_profile():
    if active_state["current_profile"] is None:
        raise HTTPException(status_code=404, detail="No active telemetry dataset loaded.")
    return active_state["current_profile"]

@app.post("/api/telemetry/aggregate")
def aggregate_telemetry_windows(window_size: str = "30s"):
    """Aggregates uploaded raw telemetry into time-window features."""
    if active_state["current_df"] is None:
        raise HTTPException(status_code=400, detail="No telemetry uploaded.")

    df = active_state["current_df"]
    detected_type = TelemetryParser.detect_telemetry_type(df)

    # Check if already aggregated
    if "p95_latency" in df.columns and "cpu_util" in df.columns:
        return {
            "message": "Dataset already contains aggregated time-window features.",
            "record_count": len(df),
            "columns": list(df.columns)
        }

    logs_df = df if detected_type == "logs" else None
    metrics_df = df if detected_type == "metrics" else None
    traces_df = df if detected_type == "traces" else None
    events_df = df if detected_type == "events" else None

    aggregated_df = pipeline.aggregate_multi_source_telemetry(
        logs_df=logs_df,
        metrics_df=metrics_df,
        traces_df=traces_df,
        events_df=events_df,
        window_size=window_size
    )

    active_state["current_df"] = aggregated_df
    active_state["current_profile"] = TelemetryParser.inspect_data_quality(
        aggregated_df, f"aggregated_{active_state['current_dataset']}"
    )

    return {
        "message": f"Successfully aggregated into {len(aggregated_df)} time-window feature records.",
        "profile": active_state["current_profile"]
    }


# -------------------------------------------------------------
# Dataset Labeling Endpoints
# -------------------------------------------------------------
@app.post("/api/labeling/rules")
def apply_rule_assisted_labeling(request: RuleLabelRequest):
    if active_state["current_df"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload or load a dataset first.")

    custom_rules = request.model_dump()
    labeled_df = TelemetryLabelingEngine.apply_rule_assisted_labels(
        active_state["current_df"], custom_rules=custom_rules
    )
    active_state["current_df"] = labeled_df

    label_distribution = labeled_df["label"].value_counts().to_dict()

    return {
        "message": "Rule-assisted labeling applied successfully.",
        "label_distribution": label_distribution,
        "sample_preview": labeled_df[["window", "service", "label", "rule_reason"]].head(10).to_dict(orient="records")
    }

@app.post("/api/labeling/manual")
def apply_manual_label(request: ManualLabelRequest):
    if active_state["current_df"] is None:
        raise HTTPException(status_code=400, detail="No active dataset.")
    if request.index < 0 or request.index >= len(active_state["current_df"]):
        raise HTTPException(status_code=400, detail="Index out of bounds.")

    active_state["current_df"].at[request.index, "label"] = request.label
    return {"message": f"Record {request.index} label updated to '{request.label}'."}

@app.post("/api/labeling/unsupervised")
def run_unsupervised_detection(contamination: float = 0.05):
    if active_state["current_df"] is None:
        raise HTTPException(status_code=400, detail="No active dataset.")

    feature_cols = TelemetryFeaturePipeline.FEATURE_COLUMNS
    res_df, summary = TelemetryLabelingEngine.run_unsupervised_anomaly_detection(
        active_state["current_df"], feature_columns=feature_cols, contamination=contamination
    )
    active_state["current_df"] = res_df
    return summary


# -------------------------------------------------------------
# Machine Learning Training & Evaluation Endpoints
# -------------------------------------------------------------
@app.post("/api/model/train")
def train_machine_learning_models(request: TrainRequest):
    if active_state["current_df"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    df = active_state["current_df"].copy()
    if "label" not in df.columns:
        # Apply rule-assisted labels if missing
        df = TelemetryLabelingEngine.apply_rule_assisted_labels(df)
        active_state["current_df"] = df

    try:
        report = ml_engine.train_models(
            df=df,
            label_col="label",
            test_size=request.test_size or 0.25,
            split_method=request.split_method or "stratified",
            n_estimators=request.n_estimators or 120,
            max_depth=request.max_depth or 6,
            learning_rate=request.learning_rate or 0.1
        )
        # Initialize XAI explainer with newly trained model
        xai_engine.initialize_explainer()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/model/models")
def list_trained_models():
    return ml_engine.list_saved_models()

@app.post("/api/model/load")
def load_trained_model(model_id: str = Query(...)):
    success = ml_engine.load_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    xai_engine.initialize_explainer()
    return {"message": f"Successfully loaded model '{model_id}'", "metadata": ml_engine.model_metadata}


# -------------------------------------------------------------
# Telemetry Inference & Anomaly Detection Endpoints
# -------------------------------------------------------------
@app.post("/api/inference/detect")
def run_telemetry_inference(request: Optional[InferenceRequest] = None):
    if ml_engine.xgb_model is None:
        # Auto-train if dataset exists
        if active_state["current_df"] is not None:
            train_machine_learning_models(TrainRequest())
        else:
            raise HTTPException(status_code=400, detail="No model trained or loaded.")

    if request and request.records:
        inference_df = pd.DataFrame(request.records)
    elif active_state["current_df"] is not None:
        inference_df = active_state["current_df"]
    else:
        raise HTTPException(status_code=400, detail="No telemetry provided for inference.")

    results = ml_engine.predict(inference_df)
    active_state["last_prediction"] = results

    # Count anomaly summary
    anomalies = [r for r in results if r["is_anomaly"]]
    normal = [r for r in results if not r["is_anomaly"]]

    return {
        "total_records": len(results),
        "anomalies_detected": len(anomalies),
        "normal_records": len(normal),
        "anomaly_rate": round(len(anomalies) / max(len(results), 1), 4),
        "predictions": results
    }


# -------------------------------------------------------------
# Explainable AI (SHAP) Endpoints
# -------------------------------------------------------------
@app.get("/api/explain/global")
def get_global_shap_explanation():
    if active_state["current_df"] is None or ml_engine.xgb_model is None:
        raise HTTPException(status_code=400, detail="Model and dataset required for SHAP.")
    try:
        return xai_engine.explain_global(active_state["current_df"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")

@app.get("/api/explain/instance")
def get_instance_shap_explanation(index: int = 0):
    if active_state["current_df"] is None or ml_engine.xgb_model is None:
        raise HTTPException(status_code=400, detail="Model and dataset required.")
    if index < 0 or index >= len(active_state["current_df"]):
        raise HTTPException(status_code=400, detail="Index out of range.")

    try:
        row = active_state["current_df"].iloc[index]
        return xai_engine.explain_instance(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation error: {str(e)}")


# -------------------------------------------------------------
# Root Cause Analysis (RCA) & Recommendation Endpoints
# -------------------------------------------------------------
@app.post("/api/rca/diagnose")
def run_root_cause_diagnosis(request: DiagnoseRequest):
    if active_state["current_df"] is None:
        raise HTTPException(status_code=400, detail="No active dataset.")

    idx = min(max(0, request.record_index or 0), len(active_state["current_df"]) - 1)
    row = active_state["current_df"].iloc[idx]
    srv = str(row.get("service", request.service or "ts-order-service"))

    # Obtain SHAP explanation for this record
    shap_data = {}
    try:
        shap_data = xai_engine.explain_instance(row)
    except Exception:
        shap_data = {
            "top_increasing_features": [{"feature": "p95_latency", "value": float(row.get("p95_latency", 50.0))}],
            "natural_language_explanation": "Telemetry features exceeded baselines."
        }

    # Extract metrics dict
    metric_feats = {col: round(float(row.get(col, 0.0)), 2) for col in TelemetryFeaturePipeline.FEATURE_COLUMNS}

    # Execute Root Cause Analysis
    diagnosis = rca_engine.diagnose_anomaly(
        predicted_category=request.predicted_category or str(row.get("label", "DATABASE_FAILURE")),
        affected_service=srv,
        shap_explanation=shap_data,
        raw_logs=active_state["raw_logs"],
        raw_traces=active_state["raw_traces"],
        raw_events=active_state["raw_events"],
        telemetry_metrics=metric_feats
    )

    return diagnosis


# -------------------------------------------------------------
# Static Dashboard Mounting
# -------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_dashboard():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "Frontend is initializing..."})
    return FileResponse(index_file)
