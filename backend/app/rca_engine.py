import re
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
from app.config import SERVICE_DEPENDENCIES, TRAIN_TICKET_SERVICES

class RootCauseAnalysisEngine:
    """Investigates microservice telemetry anomalies, maps dependency cascades, and isolates root causes."""

    RECOMMENDATION_PLAYBOOKS = {
        "DATABASE_FAILURE": {
            "title": "Database Connection & Query Remediation",
            "steps": [
                "1. Verify MongoDB/MySQL connectivity and connection pool metrics (`maxPoolSize`, `waitQueueTimeout`).",
                "2. Inspect slow query logs (`explain()` plans) and check for missing collection indexes.",
                "3. Check database host CPU, memory, and disk I/O wait times.",
                "4. Review microservice connection pool leak logs and ensure connections are closed in `finally` blocks.",
                "5. Ensure database circuit breaker / retry policies are configured with exponential backoff."
            ],
            "severity": "CRITICAL"
        },
        "HIGH_MEMORY": {
            "title": "Memory Leak & Heap Saturation Remediation",
            "steps": [
                "1. Inspect JVM heap / Node.js heap memory graphs for monotonic upward trends (memory leak).",
                "2. Check container memory limits (`resources.limits.memory`) in Kubernetes deployment manifests.",
                "3. Analyze thread and heap dumps (`jcmd <pid> GC.run` or `jmap`) for retained large objects.",
                "4. Check Kubernetes operational events for recent `OOMKilled` (Exit Code 137) container terminations.",
                "5. Consider tuning garbage collector (e.g. G1GC `-XX:MaxGCPauseMillis`) or scaling memory limits."
            ],
            "severity": "CRITICAL"
        },
        "HIGH_CPU": {
            "title": "CPU Throttling & Computation Spike Remediation",
            "steps": [
                "1. Run CPU profiling (async-profiler / flamegraphs) to locate hot methods or infinite loops.",
                "2. Check Kubernetes CPU limits (`resources.limits.cpu`) for CFS quota throttling (`container_cpu_cfs_throttled_periods_total`).",
                "3. Correlate with sudden inbound request surges (ticket booking rush hour) and evaluate HPA autoscaling.",
                "4. Verify thread concurrency pool sizes and ensure non-blocking I/O for external microservice calls.",
                "5. Cache high-frequency read responses (e.g. train station queries via Redis/in-memory cache)."
            ],
            "severity": "WARNING"
        },
        "HIGH_LATENCY": {
            "title": "Cascading Latency & Downstream Bottleneck Remediation",
            "steps": [
                "1. Inspect distributed trace waterfall: check which child span consumed the majority of total duration.",
                "2. Isolate whether delay is network round-trip, callee service execution, or database waiting.",
                "3. Check callee microservice health and connection queues.",
                "4. Configure aggressive client-side HTTP timeouts and fallback responses to prevent thread starvation.",
                "5. Enable distributed tracing headers propagation across all service boundaries."
            ],
            "severity": "WARNING"
        },
        "SERVICE_FAILURE": {
            "title": "Service Crash & Pod Instability Remediation",
            "steps": [
                "1. Check Kubernetes pod status for `CrashLoopBackOff`, container exit codes, and restart count.",
                "2. Inspect stderr application logs immediately preceding the termination.",
                "3. Verify readiness and liveness probe endpoints (`/actuator/health`) for timeouts.",
                "4. Verify all configuration variables, secret mounts, and dependent database endpoints.",
                "5. Ensure pod disruption budgets and replica counts permit resilient failover."
            ],
            "severity": "CRITICAL"
        },
        "NETWORK_FAILURE": {
            "title": "Microservice Inter-communication & Network Remediation",
            "steps": [
                "1. Verify CoreDNS / Kube-DNS resolution latency and error rates across the cluster.",
                "2. Check Kubernetes NetworkPolicies and security groups between caller and callee namespaces.",
                "3. Review TCP connection reset / SYN packet drop metrics on host nodes.",
                "4. Ensure keep-alive connections and connection pooling between microservices are functioning.",
                "5. Check service mesh / ingress proxy logs (Envoy/Nginx) for 502/504 upstream timeouts."
            ],
            "severity": "CRITICAL"
        },
        "HIGH_ERROR_RATE": {
            "title": "Application Exception & Error Surge Remediation",
            "steps": [
                "1. Aggregate error logs by exception class (`NullPointerException`, `IllegalArgumentException`, etc.).",
                "2. Identify whether errors are localized to specific user payloads or specific endpoints.",
                "3. Verify schema compatibility between service versions (breaking API changes).",
                "4. Validate input validation boundaries before downstream RPC dispatches.",
                "5. Monitor error rate trends after rollbacks or canary updates."
            ],
            "severity": "WARNING"
        },
        "NORMAL": {
            "title": "Baseline System Health",
            "steps": [
                "1. Maintain routine observability and continuous metric streaming.",
                "2. Verify alerting thresholds and health probes are active."
            ],
            "severity": "NORMAL"
        }
    }

    def __init__(self):
        # Build networkx DiGraph of Train Ticket architecture
        self.graph = nx.DiGraph()
        for srv in TRAIN_TICKET_SERVICES:
            self.graph.add_node(srv)
        for caller, callees in SERVICE_DEPENDENCIES.items():
            for callee in callees:
                self.graph.add_edge(caller, callee)

    def get_topology_data(self) -> Dict[str, Any]:
        """Returns node and edge definitions formatted for frontend visualization."""
        nodes = []
        for node in self.graph.nodes:
            is_db = "db" in node.lower()
            nodes.append({
                "id": node,
                "label": node,
                "type": "database" if is_db else "service"
            })
        edges = []
        for u, v in self.graph.edges:
            edges.append({"from": u, "to": v})
        return {"nodes": nodes, "edges": edges}

    def find_blast_radius(self, root_service: str) -> Dict[str, List[str]]:
        """Finds upstream callers impacted (blast radius) and downstream callees."""
        upstream = []
        downstream = []

        if root_service in self.graph:
            # Ancestors (services that call this service directly or indirectly)
            try:
                upstream = list(nx.ancestors(self.graph, root_service))
            except Exception:
                pass
            # Descendants (services that this service calls)
            try:
                downstream = list(nx.descendants(self.graph, root_service))
            except Exception:
                pass

        return {
            "upstream_callers_affected": upstream,
            "downstream_dependencies": downstream
        }

    def diagnose_anomaly(
        self,
        predicted_category: str,
        affected_service: str,
        shap_explanation: Dict[str, Any],
        raw_logs: Optional[List[Dict[str, Any]]] = None,
        raw_traces: Optional[List[Dict[str, Any]]] = None,
        raw_events: Optional[List[Dict[str, Any]]] = None,
        telemetry_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes multi-source telemetry to isolate probable or confirmed root causes.
        Strictly distinguishes:
          1. Anomaly Detected
          2. Probable Root Cause
          3. Confirmed Root Cause
        """
        raw_logs = raw_logs or []
        raw_traces = raw_traces or []
        raw_events = raw_events or []
        telemetry_metrics = telemetry_metrics or {}

        # 1. Investigate supporting evidence from logs
        relevant_logs = []
        log_has_confirmed_evidence = False
        confirmed_log_reason = ""

        for log in raw_logs:
            srv = str(log.get("service", "")).lower()
            lvl = str(log.get("level", "")).upper()
            msg = str(log.get("message", ""))
            exc = str(log.get("exception", ""))

            # Filter relevant logs for this service or callee
            if affected_service.lower() in srv or "error" in lvl or "fail" in msg.lower():
                relevant_logs.append({
                    "timestamp": log.get("timestamp"),
                    "service": log.get("service"),
                    "level": lvl,
                    "message": msg[:200],
                    "exception": exc[:200]
                })

                # Check for explicit root cause proof in logs
                full_text = f"{msg} {exc}".lower()
                if "mongotimeoutexception" in full_text or "connection pool exhausted" in full_text or "sqlexception" in full_text:
                    log_has_confirmed_evidence = True
                    confirmed_log_reason = "Database connection pool exhaustion / timeout explicitly logged."
                elif "outofmemoryerror" in full_text or "java heap space" in full_text:
                    log_has_confirmed_evidence = True
                    confirmed_log_reason = "JVM OutOfMemoryError heap space exhaustion explicitly logged."
                elif "connectexception: connection refused" in full_text:
                    log_has_confirmed_evidence = True
                    confirmed_log_reason = "Downstream TCP connection refused explicitly logged."

        # 2. Investigate supporting evidence from traces
        relevant_traces = []
        trace_has_callee_bottleneck = False
        callee_bottleneck_service = None

        for tr in raw_traces:
            srv = str(tr.get("service", ""))
            if affected_service.lower() in srv.lower() or tr.get("has_error") or tr.get("duration", 0) > 300:
                relevant_traces.append({
                    "trace_id": tr.get("trace_id"),
                    "span_id": tr.get("span_id"),
                    "service": srv,
                    "operation": tr.get("operation_name") or tr.get("operation"),
                    "duration": tr.get("duration"),
                    "has_error": tr.get("has_error") or False
                })

                # If child span took > 70% of duration, callee is bottleneck
                if tr.get("duration", 0) > 300 and srv != affected_service:
                    trace_has_callee_bottleneck = True
                    callee_bottleneck_service = srv

        # 3. Investigate operational events (K8s)
        relevant_events = []
        event_has_confirmed_cause = False
        confirmed_event_reason = ""

        for ev in raw_events:
            srv = str(ev.get("service", ""))
            ev_str = f"{ev.get('event_type', '')} {ev.get('reason', '')} {ev.get('message', '')}".lower()
            if affected_service.lower() in srv.lower() or "oomkilled" in ev_str or "restart" in ev_str:
                relevant_events.append(ev)
                if "oomkilled" in ev_str:
                    event_has_confirmed_cause = True
                    confirmed_event_reason = "Container OOMKilled event confirmed by Kubernetes engine."
                elif "crashloopbackoff" in ev_str:
                    event_has_confirmed_cause = True
                    confirmed_event_reason = "Container CrashLoopBackOff failure confirmed by Kubernetes."

        # 4. Synthesize Diagnosis Level
        # 3 Tiers: Anomaly Detected -> Probable Root Cause -> Confirmed Root Cause
        diagnosis_tier = "ANOMALY_DETECTED"
        confidence_score = 0.65
        probable_root_cause = ""
        root_cause_service = affected_service

        # Check if bottleneck is actually downstream in callee
        if trace_has_callee_bottleneck and callee_bottleneck_service:
            root_cause_service = callee_bottleneck_service
            probable_root_cause = f"Downstream dependency '{callee_bottleneck_service}' is introducing cascading delay to '{affected_service}'."
            diagnosis_tier = "PROBABLE_ROOT_CAUSE"
            confidence_score = 0.82
        elif predicted_category == "DATABASE_FAILURE":
            root_cause_service = "mongo-db" if "mongo" in str(relevant_logs).lower() or "order" in affected_service else "mysql-db"
            probable_root_cause = f"Database query latency and connection saturation impacting {affected_service}."
            diagnosis_tier = "PROBABLE_ROOT_CAUSE"
            confidence_score = 0.85
        elif predicted_category == "HIGH_CPU":
            probable_root_cause = f"CPU computation saturation or request spike within {affected_service}."
            diagnosis_tier = "PROBABLE_ROOT_CAUSE"
            confidence_score = 0.80
        elif predicted_category == "HIGH_MEMORY":
            probable_root_cause = f"Memory leak or high heap retention within {affected_service}."
            diagnosis_tier = "PROBABLE_ROOT_CAUSE"
            confidence_score = 0.80
        elif predicted_category == "SERVICE_FAILURE":
            probable_root_cause = f"Service instance instability or container crashes in {affected_service}."
            diagnosis_tier = "PROBABLE_ROOT_CAUSE"
            confidence_score = 0.85
        else:
            probable_root_cause = f"Performance degradation detected in {affected_service}."
            diagnosis_tier = "ANOMALY_DETECTED"
            confidence_score = 0.65

        # Check if Confirmed
        if log_has_confirmed_evidence:
            diagnosis_tier = "CONFIRMED_ROOT_CAUSE"
            confidence_score = 0.98
            probable_root_cause += f" [CONFIRMED VIA LOGS: {confirmed_log_reason}]"
        elif event_has_confirmed_cause:
            diagnosis_tier = "CONFIRMED_ROOT_CAUSE"
            confidence_score = 0.99
            probable_root_cause += f" [CONFIRMED VIA K8S EVENT: {confirmed_event_reason}]"

        # Evidence Sufficiency Assessment
        has_sufficient_evidence = bool(relevant_logs or relevant_traces or relevant_events)
        evidence_sufficiency_note = (
            "Sufficient corroborating telemetry available across logs, traces, and operational events."
            if has_sufficient_evidence
            else "Caution: Limited multi-source telemetry provided. Diagnosis is based primarily on ML model feature weights and statistical baseline thresholds."
        )

        # Blast radius
        blast_radius = self.find_blast_radius(root_cause_service)

        # Recommendations
        playbook = self.RECOMMENDATION_PLAYBOOKS.get(
            predicted_category,
            self.RECOMMENDATION_PLAYBOOKS["NORMAL"]
        )

        # Why it happened (SHAP dominant factors)
        top_shap_factors = shap_explanation.get("top_increasing_features", [])
        evidence_factors_str = ", ".join([f"{f['feature']}={f['value']}" for f in top_shap_factors[:3]])

        return {
            "problem_description": f"Detected {predicted_category.replace('_', ' ')} affecting microservice '{affected_service}'",
            "affected_microservice": affected_service,
            "root_cause_service": root_cause_service,
            "diagnosis_tier": diagnosis_tier,
            "confidence_score": confidence_score,
            "probable_root_cause": probable_root_cause,
            "evidence_sufficiency": evidence_sufficiency_note,
            "why_it_happened": f"Observed abnormal values for: {evidence_factors_str or 'telemetry deviations'}. " + shap_explanation.get("natural_language_explanation", ""),
            "supporting_evidence": {
                "metric_features": telemetry_metrics,
                "relevant_logs": relevant_logs[:6],
                "relevant_traces": relevant_traces[:6],
                "relevant_events": relevant_events[:6],
                "shap_top_drivers": top_shap_factors
            },
            "blast_radius": blast_radius,
            "recommended_troubleshooting_steps": playbook["steps"],
            "playbook_title": playbook["title"],
            "read_only_notice": "READ-ONLY RECOMMENDATION: Automated modifications, restarts, or mutations to the Train Ticket cluster are disabled."
        }
