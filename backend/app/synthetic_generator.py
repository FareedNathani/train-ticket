import os
import json
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from app.config import SYNTHETIC_DIR, TRAIN_TICKET_SERVICES

class TrainTicketBenchmarkGenerator:
    """Generates high-fidelity Train Ticket telemetry datasets across logs, metrics, traces, and events."""

    @staticmethod
    def generate_benchmark_datasets(num_windows: int = 150) -> Dict[str, str]:
        """Generates synchronized logs, metrics, traces, and operational events with real failure injections."""
        base_time = datetime.now() - timedelta(minutes=num_windows * 2)

        logs_data = []
        metrics_data = []
        traces_data = []
        events_data = []
        ground_truth_windows = []

        services = [
            "ts-ui-dashboard",
            "ts-order-service",
            "ts-order-other-service",
            "ts-travel-service",
            "ts-ticketinfo-service",
            "ts-station-service",
            "ts-preserve-service",
            "ts-inside-payment-service",
            "ts-route-service",
            "ts-user-service",
            "mongo-db"
        ]

        trace_counter = 1000

        for w in range(num_windows):
            current_window_time = base_time + timedelta(seconds=w * 30)
            iso_time = current_window_time.isoformat()

            # Determine if this window has an injected scenario
            scenario = "NORMAL"
            target_service = random.choice(services)

            if 20 <= w < 40:
                scenario = "DATABASE_FAILURE"
                target_service = "ts-order-service"
            elif 55 <= w < 75:
                scenario = "HIGH_MEMORY"
                target_service = "ts-ticketinfo-service"
            elif 90 <= w < 105:
                scenario = "HIGH_CPU"
                target_service = "ts-preserve-service"
            elif 120 <= w < 135:
                scenario = "NETWORK_FAILURE"
                target_service = "ts-order-service"

            # 1. Generate Metrics for each service in this window
            for srv in services:
                is_target = (srv == target_service) and (scenario != "NORMAL")

                if is_target:
                    if scenario == "DATABASE_FAILURE":
                        cpu = random.uniform(35.0, 55.0)
                        mem = random.uniform(50.0, 65.0)
                        lat = random.uniform(450.0, 1200.0)
                        err_rate = random.uniform(0.18, 0.45)
                        throughput = random.uniform(25.0, 40.0)
                        db_dur = random.uniform(380.0, 950.0)
                    elif scenario == "HIGH_MEMORY":
                        cpu = random.uniform(40.0, 60.0)
                        mem = min(99.0, 70.0 + (w - 55) * 1.5)
                        lat = random.uniform(350.0, 750.0)
                        err_rate = random.uniform(0.10, 0.30)
                        throughput = random.uniform(30.0, 50.0)
                        db_dur = random.uniform(15.0, 30.0)
                    elif scenario == "HIGH_CPU":
                        cpu = random.uniform(88.0, 98.5)
                        mem = random.uniform(55.0, 70.0)
                        lat = random.uniform(250.0, 550.0)
                        err_rate = random.uniform(0.05, 0.15)
                        throughput = random.uniform(180.0, 260.0)
                        db_dur = random.uniform(10.0, 25.0)
                    elif scenario == "NETWORK_FAILURE":
                        cpu = random.uniform(20.0, 40.0)
                        mem = random.uniform(40.0, 55.0)
                        lat = random.uniform(800.0, 2000.0)
                        err_rate = random.uniform(0.35, 0.65)
                        throughput = random.uniform(10.0, 25.0)
                        db_dur = random.uniform(10.0, 20.0)
                    else:
                        cpu, mem, lat, err_rate, throughput, db_dur = 25.0, 45.0, 45.0, 0.0, 80.0, 10.0
                else:
                    # Normal baseline traffic
                    cpu = random.uniform(15.0, 42.0)
                    mem = random.uniform(35.0, 58.0)
                    lat = random.uniform(20.0, 70.0)
                    err_rate = random.uniform(0.0, 0.02)
                    throughput = random.uniform(70.0, 110.0)
                    db_dur = random.uniform(5.0, 18.0)

                metrics_data.append({
                    "timestamp": iso_time,
                    "service": srv,
                    "cpu_util": round(cpu, 2),
                    "memory_util": round(mem, 2),
                    "latency": round(lat, 2),
                    "p95_latency": round(lat * 1.3, 2),
                    "p99_latency": round(lat * 1.7, 2),
                    "error_rate": round(err_rate, 4),
                    "throughput": round(throughput, 1),
                    "db_duration_avg": round(db_dur, 2)
                })

                # Ground truth record for training
                ground_truth_windows.append({
                    "window": iso_time,
                    "service": srv,
                    "cpu_util": round(cpu, 2),
                    "memory_util": round(mem, 2),
                    "avg_latency": round(lat, 2),
                    "p95_latency": round(lat * 1.3, 2),
                    "p99_latency": round(lat * 1.7, 2),
                    "error_count": int(err_rate * 50),
                    "warning_count": int(err_rate * 20),
                    "error_rate": round(err_rate, 4),
                    "request_throughput": round(throughput, 1),
                    "failed_span_count": int(err_rate * 15) if is_target else 0,
                    "trace_duration_avg": round(lat, 2),
                    "db_duration_avg": round(db_dur, 2),
                    "restart_count": 1 if (is_target and scenario == "HIGH_MEMORY" and mem > 94.0) else 0,
                    "downstream_delay_ratio": round(max(0.0, (lat - db_dur) / (lat + 1e-5)), 2),
                    "affected_service_count": len(services) if is_target else 1,
                    "label": scenario if is_target else "NORMAL"
                })

            # 2. Generate Logs for this window
            trace_counter += 1
            trace_id = f"trc-{trace_counter:06d}"

            if scenario == "DATABASE_FAILURE":
                logs_data.append({
                    "timestamp": iso_time,
                    "service": "ts-order-service",
                    "level": "ERROR",
                    "message": "com.mongodb.MongoTimeoutException: Timed out after 30000 ms while waiting for a server that matches ReadPreference.primary(). Connection pool exhausted.",
                    "trace_id": trace_id,
                    "span_id": f"spn-{random.randint(100, 999)}",
                    "exception": "com.mongodb.MongoTimeoutException: connection pool exhausted at com.mongodb.internal.connection.BaseCluster.selectServer(BaseCluster.java:101)"
                })
            elif scenario == "HIGH_MEMORY":
                logs_data.append({
                    "timestamp": iso_time,
                    "service": "ts-ticketinfo-service",
                    "level": "ERROR" if w > 65 else "WARN",
                    "message": "java.lang.OutOfMemoryError: Java heap space. Garbage collection overhead limit exceeded." if w > 65 else "High heap memory utilization (>85%). GC pause duration 1820ms.",
                    "trace_id": trace_id,
                    "span_id": f"spn-{random.randint(100, 999)}",
                    "exception": "java.lang.OutOfMemoryError: Java heap space" if w > 65 else ""
                })
            elif scenario == "NETWORK_FAILURE":
                logs_data.append({
                    "timestamp": iso_time,
                    "service": "ts-order-service",
                    "level": "ERROR",
                    "message": "org.springframework.web.client.ResourceAccessException: I/O error on POST request for 'http://ts-inside-payment-service:8080/pay': Connect timed out",
                    "trace_id": trace_id,
                    "span_id": f"spn-{random.randint(100, 999)}",
                    "exception": "java.net.SocketTimeoutException: Read timed out"
                })
            else:
                logs_data.append({
                    "timestamp": iso_time,
                    "service": "ts-order-service",
                    "level": "INFO",
                    "message": f"Order processed successfully for user {random.randint(1001, 1099)}. Ticket confirmed.",
                    "trace_id": trace_id,
                    "span_id": f"spn-{random.randint(100, 999)}",
                    "exception": ""
                })

            # 3. Generate Distributed Traces for this window
            is_err_trace = (scenario in ["DATABASE_FAILURE", "NETWORK_FAILURE"])
            root_span_id = f"spn-root-{w}"
            order_span_id = f"spn-order-{w}"
            db_span_id = f"spn-db-{w}"

            traces_data.append({
                "trace_id": trace_id,
                "span_id": root_span_id,
                "parent_span_id": None,
                "service": "ts-ui-dashboard",
                "operation_name": "/api/v1/orders/preserve",
                "timestamp": iso_time,
                "duration": 650.0 if is_err_trace else 48.0,
                "has_error": is_err_trace,
                "status_code": 500 if is_err_trace else 200
            })
            traces_data.append({
                "trace_id": trace_id,
                "span_id": order_span_id,
                "parent_span_id": root_span_id,
                "service": "ts-order-service",
                "operation_name": "createOrder",
                "timestamp": iso_time,
                "duration": 580.0 if is_err_trace else 35.0,
                "has_error": is_err_trace,
                "status_code": 500 if is_err_trace else 200
            })
            traces_data.append({
                "trace_id": trace_id,
                "span_id": db_span_id,
                "parent_span_id": order_span_id,
                "service": "mongo-db",
                "operation_name": "db.orders.insert",
                "timestamp": iso_time,
                "duration": 520.0 if scenario == "DATABASE_FAILURE" else 12.0,
                "has_error": (scenario == "DATABASE_FAILURE"),
                "status_code": 500 if scenario == "DATABASE_FAILURE" else 200
            })

            # 4. Generate Operational Events (K8s)
            if scenario == "HIGH_MEMORY" and w == 70:
                events_data.append({
                    "timestamp": iso_time,
                    "service": "ts-ticketinfo-service",
                    "event_type": "OOMKilled",
                    "reason": "ContainerKilled",
                    "message": "Pod ts-ticketinfo-service-7f8d9b-x92zk container ts-ticketinfo was terminated: OOMKilled (Exit Code 137). Limit: 1024Mi."
                })
            elif scenario == "SERVICE_FAILURE" and w % 20 == 0:
                events_data.append({
                    "timestamp": iso_time,
                    "service": "ts-order-other-service",
                    "event_type": "PodRestart",
                    "reason": "CrashLoopBackOff",
                    "message": "Back-off 5m0s restarting failed container in pod ts-order-other-service-5d6c8b-k281l."
                })

        # Save to disk as files
        paths = {}

        # 1. Logs CSV
        logs_df = pd.DataFrame(logs_data)
        logs_path = SYNTHETIC_DIR / "train_ticket_logs.csv"
        logs_df.to_csv(logs_path, index=False)
        paths["logs"] = str(logs_path)

        # 2. Metrics CSV
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = SYNTHETIC_DIR / "train_ticket_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        paths["metrics"] = str(metrics_path)

        # 3. Traces JSONL
        traces_path = SYNTHETIC_DIR / "train_ticket_traces.jsonl"
        with open(traces_path, "w") as f:
            for tr in traces_data:
                f.write(json.dumps(tr) + "\n")
        paths["traces"] = str(traces_path)

        # 4. Events JSON
        events_path = SYNTHETIC_DIR / "train_ticket_events.json"
        with open(events_path, "w") as f:
            json.dump(events_data, f, indent=2)
        paths["events"] = str(events_path)

        # 5. Pre-aggregated Ground Truth Dataset for Instant Training
        gt_df = pd.DataFrame(ground_truth_windows)
        gt_path = SYNTHETIC_DIR / "train_ticket_aggregated_benchmark.csv"
        gt_df.to_csv(gt_path, index=False)
        paths["benchmark"] = str(gt_path)

        return paths
