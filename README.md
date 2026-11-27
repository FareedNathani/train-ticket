# AI-Powered Microservice Observability, Anomaly Detection & Root Cause Analysis System

> **Target Architecture:** [FudanSELab Train Ticket](https://github.com/FudanSELab/train-ticket.git) (~41 Microservices: Java Spring Boot, Node.js, Python, Go, MongoDB, MySQL, Kubernetes).

---

## 1. Overview
This system is a production-grade, AI-driven microservice observability platform. It allows engineers to upload multi-source telemetry collected from Train Ticket microservices without requiring the heavy Train Ticket cluster (31GB RAM) to be deployed on local workstations.

The platform provides:
- **Multi-Source Telemetry Ingestion:** Application Logs (Java, Node.js, Python, Go), Prometheus Metrics, Distributed Traces (Jaeger/OTel), and Operational Events (Kubernetes).
- **Data Quality Profiling & Schema Mapping:** Missing value counters, duplicate detection, schema standardization, and preview.
- **Dataset Labeling Studio:** Heuristic rule-assisted labeling, manual tagging, label import/export, and unsupervised Isolation Forest outlier detection.
- **Dual ML Training & True Evaluation:** **XGBoost** (Primary) vs **Random Forest** (Baseline), stratified and temporal train/test splitting, confusion matrix heatmap, and genuine metrics (Accuracy, Precision, Recall, F1).
- **SHAP Explainable AI:** Global mean |SHAP| feature importances, local waterfall feature attribution, and natural-language grounded explanations.
- **Root Cause Analysis (RCA) & Service Dependency Graph:** 41-microservice topology visualizer, blast radius mapping, multi-source evidence correlation, and 3-tier diagnosis (**Anomaly Detected** -> **Probable Root Cause** -> **Confirmed Root Cause**).
- **Intelligent Remediation Engine:** Context-aware, strictly read-only troubleshooting playbooks.

---

## 2. Quick Start

### Start the Server
```bash
python run_server.py
```

### Access Dashboards
- **Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **FastAPI Interactive API Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Run Automated Tests
```bash
pytest -v tests/test_system.py
```

---

## 3. Telemetry Formats Supported
- **CSV:** Tabular metrics, logs, or traces with column headers.
- **JSON / JSONL:** Structured application logs, Jaeger/Zipkin spans, and Kubernetes pod events.
- **TXT:** Raw multiline logs from Java Spring Boot, Go, or Python microservices.

---

## 4. Benchmark Telemetry Dataset
For immediate testing and verification without manual data collection, click the **"Load Pre-packaged Train Ticket Benchmark"** button on Tab 1 of the dashboard. This loads a synchronized benchmark with realistic failure injections:
1. `DATABASE_FAILURE`: MongoDB connection pool exhaustion on `ts-order-service`.
2. `HIGH_MEMORY`: JVM heap saturation and Kubernetes `OOMKilled` event on `ts-ticketinfo-service`.
3. `HIGH_CPU`: CPU computation surge on `ts-preserve-service`.
4. `NETWORK_FAILURE`: Downstream RPC timeout between `ts-order-service` and `ts-payment-service`.
5. `NORMAL`: Baseline healthy microservice operations.
