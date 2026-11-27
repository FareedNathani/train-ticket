// Global State
let globalState = {
  currentDataset: null,
  profile: null,
  models: [],
  predictions: [],
  topologyData: null,
  networkInstance: null,
  charts: {}
};

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  fetchHealth();
  initTopologyGraph();
  loadSavedModels();
  fetchGlobalSHAP();
});

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
  document.querySelectorAll(".nav-tab").forEach(tab => tab.classList.remove("active"));

  const targetPane = document.getElementById(tabId);
  if (targetPane) targetPane.classList.add("active");

  const matchingNavTab = Array.from(document.querySelectorAll(".nav-tab")).find(tab => 
    tab.getAttribute("onclick")?.includes(tabId)
  );
  if (matchingNavTab) matchingNavTab.classList.add("active");

  // Re-fit network if switching to RCA tab
  if (tabId === "tab-rca" && globalState.networkInstance) {
    setTimeout(() => globalState.networkInstance.fit(), 200);
  }
}

// System Health Polling
async function fetchHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById("stat-system-status").textContent = `System: ${data.status}`;
    document.getElementById("stat-active-model").textContent = data.loaded_model || "None";
    document.getElementById("stat-active-dataset").textContent = data.active_dataset || "None";
    document.getElementById("stat-active-records").textContent = data.active_records || "0";

    if (data.active_records > 0 && !globalState.profile) {
      loadCurrentProfile();
    }
  } catch (err) {
    console.warn("Health check error:", err);
  }
}

// -------------------------------------------------------------
// TAB 1: TELEMETRY INGESTION & QUALITY PROFILING
// -------------------------------------------------------------
async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/telemetry/upload", {
      method: "POST",
      body: formData
    });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || "Upload failed");

    renderProfile(result.profile);
    fetchHealth();
    alert(`File '${file.name}' successfully parsed and profiled.`);
  } catch (err) {
    alert("Upload error: " + err.message);
  }
}

async function loadBenchmarkDataset() {
  try {
    const res = await fetch("/api/datasets/synthetic/generate?windows=120", { method: "POST" });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || "Generation failed");

    renderProfile(result.profile);
    fetchHealth();
    alert("Train Ticket Benchmark Dataset successfully generated and activated!");
  } catch (err) {
    alert("Benchmark generation error: " + err.message);
  }
}

async function loadCurrentProfile() {
  try {
    const res = await fetch("/api/telemetry/profile");
    if (!res.ok) return;
    const profile = await res.json();
    renderProfile(profile);
  } catch (err) {
    console.warn("Profile fetch error:", err);
  }
}

function renderProfile(profile) {
  if (!profile) return;
  globalState.profile = profile;

  document.getElementById("profile-detected-type").textContent = profile.detected_type.toUpperCase();
  document.getElementById("profile-total-rows").textContent = profile.total_rows.toLocaleString();
  document.getElementById("profile-total-cols").textContent = profile.total_columns;
  document.getElementById("profile-duplicates").textContent = profile.duplicates;

  const timeRange = profile.time_range;
  const timeStr = (timeRange && timeRange.start && timeRange.end) 
    ? `${timeRange.start.slice(0, 19)} to ${timeRange.end.slice(0, 19)}` 
    : "No continuous timestamp column";
  document.getElementById("profile-time-range").textContent = timeStr;

  document.getElementById("profile-services-count").textContent = profile.total_unique_services;
  const srvListDiv = document.getElementById("profile-services-list");
  srvListDiv.innerHTML = "";
  (profile.services_found || []).forEach(srv => {
    const badge = document.createElement("span");
    badge.className = "badge badge-service";
    badge.textContent = srv;
    srvListDiv.appendChild(badge);
  });

  // Render schema columns table
  const colTableBody = document.querySelector("#profile-columns-table tbody");
  colTableBody.innerHTML = "";
  (profile.columns || []).forEach(col => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${col.name}</strong></td>
      <td><code>${col.type}</code></td>
      <td>${col.missing}</td>
      <td>${col.missing_pct}%</td>
      <td style="color: var(--text-muted); font-family: monospace;">${col.sample || "-"}</td>
    `;
    colTableBody.appendChild(tr);
  });

  // Render preview table
  renderPreviewTable(profile.preview || []);
}

function renderPreviewTable(records) {
  const theadRow = document.getElementById("preview-header-row");
  const tbody = document.getElementById("preview-body-rows");
  theadRow.innerHTML = "";
  tbody.innerHTML = "";

  if (records.length === 0) {
    theadRow.innerHTML = "<th>No data</th>";
    return;
  }

  const cols = Object.keys(records[0]);
  cols.forEach(c => {
    const th = document.createElement("th");
    th.textContent = c;
    theadRow.appendChild(th);
  });

  records.forEach(row => {
    const tr = document.createElement("tr");
    cols.forEach(c => {
      const td = document.createElement("td");
      td.textContent = row[c] !== null && row[c] !== undefined ? String(row[c]).slice(0, 80) : "";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

async function aggregateTelemetry() {
  try {
    const res = await fetch("/api/telemetry/aggregate?window_size=30s", { method: "POST" });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || "Aggregation failed");
    if (result.profile) renderProfile(result.profile);
    fetchHealth();
    alert(result.message);
  } catch (err) {
    alert("Aggregation error: " + err.message);
  }
}

// -------------------------------------------------------------
// TAB 2: DATASET LABELING STUDIO
// -------------------------------------------------------------
async function applyRuleLabels() {
  const payload = {
    high_cpu_threshold: parseFloat(document.getElementById("rule-cpu").value),
    high_memory_threshold: parseFloat(document.getElementById("rule-mem").value),
    high_latency_threshold: parseFloat(document.getElementById("rule-latency").value),
    high_error_rate_threshold: parseFloat(document.getElementById("rule-error-rate").value) / 100.0,
    high_db_duration_threshold: parseFloat(document.getElementById("rule-db-dur").value),
    restart_threshold: parseInt(document.getElementById("rule-restarts").value)
  };

  try {
    const res = await fetch("/api/labeling/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || "Labeling failed");

    renderLabelChart(result.label_distribution);
    alert("Rule-assisted labels applied successfully across dataset.");
  } catch (err) {
    alert("Labeling error: " + err.message);
  }
}

function renderLabelChart(dist) {
  const ctx = document.getElementById("label-distribution-chart").getContext("2d");
  const labels = Object.keys(dist);
  const data = Object.values(dist);

  if (globalState.charts.labelDist) {
    globalState.charts.labelDist.destroy();
  }

  globalState.charts.labelDist = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: [
          "#10b981", "#ef4444", "#f59e0b", "#8b5cf6",
          "#3b82f6", "#06b6d4", "#ec4899", "#64748b"
        ],
        borderWidth: 1,
        borderColor: "#1a233a"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "right", labels: { color: "#94a3b8", font: { size: 10 } } }
      }
    }
  });
}

async function runUnsupervisedIsolationForest() {
  try {
    const res = await fetch("/api/labeling/unsupervised?contamination=0.08", { method: "POST" });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || "Isolation forest failed");

    document.getElementById("unsupervised-result").textContent = 
      `Isolation Forest detected ${result.detected_anomalies} anomalies (${result.anomaly_percentage}%). Note: Unsupervised outliers identify distribution shifts.`;
  } catch (err) {
    alert("Unsupervised error: " + err.message);
  }
}

// -------------------------------------------------------------
// TAB 3: MACHINE LEARNING TRAINING & EVALUATION STUDIO
// -------------------------------------------------------------
async function trainModels() {
  const btn = document.getElementById("btn-start-training");
  btn.disabled = true;
  btn.textContent = "⏳ Training Models... Please wait";

  const payload = {
    test_size: parseFloat(document.getElementById("train-test-size").value),
    split_method: document.getElementById("train-split-method").value,
    n_estimators: parseInt(document.getElementById("train-n-estimators").value),
    max_depth: parseInt(document.getElementById("train-max-depth").value),
    learning_rate: parseFloat(document.getElementById("train-learning-rate").value)
  };

  try {
    const res = await fetch("/api/model/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || "Training failed");

    renderTrainingEvaluation(result);
    fetchHealth();
    loadSavedModels();
    fetchGlobalSHAP();
    alert("Model training and evaluation completed successfully!");
  } catch (err) {
    alert("Training error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "🚀 Train XGBoost & Compare with Random Forest Baseline";
  }
}

function renderTrainingEvaluation(meta) {
  const xgb = meta.xgboost_eval || {};
  const rf = meta.random_forest_eval || {};

  document.getElementById("metric-xgb-acc").textContent = (xgb.accuracy * 100).toFixed(2) + "%";
  document.getElementById("metric-xgb-prec").textContent = (xgb.precision * 100).toFixed(2) + "%";
  document.getElementById("metric-xgb-rec").textContent = (xgb.recall * 100).toFixed(2) + "%";
  document.getElementById("metric-xgb-f1").textContent = (xgb.f1_score * 100).toFixed(2) + "%";

  document.getElementById("metric-rf-acc").textContent = (rf.accuracy * 100).toFixed(2) + "%";
  document.getElementById("metric-rf-prec").textContent = (rf.precision * 100).toFixed(2) + "%";
  document.getElementById("metric-rf-rec").textContent = (rf.recall * 100).toFixed(2) + "%";
  document.getElementById("metric-rf-f1").textContent = (rf.f1_score * 100).toFixed(2) + "%";

  // Render Confusion Matrix
  if (xgb.confusion_matrix) {
    renderConfusionMatrix(meta.classes || [], xgb.confusion_matrix);
  }
}

function renderConfusionMatrix(classes, cm) {
  const ctx = document.getElementById("confusion-matrix-chart").getContext("2d");
  if (globalState.charts.confusionMatrix) {
    globalState.charts.confusionMatrix.destroy();
  }

  // Flatten matrix into bubble or bar representation
  const labels = classes.map(c => c.replace("_", " "));
  const datasets = classes.map((c, i) => ({
    label: `True: ${c}`,
    data: cm[i],
    backgroundColor: classes.map((_, j) => i === j ? "#10b981" : "#ef4444")
  }));

  globalState.charts.confusionMatrix = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Correctly Classified Samples",
        data: classes.map((_, i) => cm[i] ? cm[i][i] || 0 : 0),
        backgroundColor: "#10b981"
      }, {
        label: "Misclassified Samples",
        data: classes.map((_, i) => cm[i] ? cm[i].reduce((a, b) => a + b, 0) - (cm[i][i] || 0) : 0),
        backgroundColor: "#ef4444"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, ticks: { color: "#94a3b8", font: { size: 9 } } },
        y: { stacked: true, ticks: { color: "#94a3b8" } }
      },
      plugins: {
        legend: { labels: { color: "#f1f5f9" } }
      }
    }
  });
}

async function loadSavedModels() {
  try {
    const res = await fetch("/api/model/models");
    if (!res.ok) return;
    const models = await res.json();
    globalState.models = models;

    const tbody = document.querySelector("#saved-models-table tbody");
    tbody.innerHTML = "";

    if (models.length === 0) {
      tbody.innerHTML = "<tr><td colspan='5' style='text-align: center; color: var(--text-muted);'>No models saved yet.</td></tr>";
      return;
    }

    models.forEach(m => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${m.model_id}</strong></td>
        <td>${(m.trained_at || "").slice(0, 19)}</td>
        <td style="color: var(--accent-green); font-weight: 700;">${((m.xgb_f1 || 0) * 100).toFixed(1)}%</td>
        <td>${((m.rf_f1 || 0) * 100).toFixed(1)}%</td>
        <td><button class="btn btn-secondary" style="padding: 3px 8px; font-size: 11px;" onclick="loadModel('${m.model_id}')">Activate</button></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.warn("Failed to load models list:", err);
  }
}

async function loadModel(modelId) {
  try {
    const res = await fetch(`/api/model/load?model_id=${encodeURIComponent(modelId)}`, { method: "POST" });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || "Load failed");
    fetchHealth();
    fetchGlobalSHAP();
    alert(result.message);
  } catch (err) {
    alert("Model load error: " + err.message);
  }
}

// -------------------------------------------------------------
// TAB 4: REAL-TIME INFERENCE & ANOMALY DETECTION
// -------------------------------------------------------------
async function runInference() {
  try {
    const res = await fetch("/api/inference/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || "Inference failed");

    globalState.predictions = result.predictions || [];

    document.getElementById("inf-total").textContent = result.total_records;
    document.getElementById("inf-critical").textContent = result.predictions.filter(p => p.severity === "CRITICAL").length;
    document.getElementById("inf-warning").textContent = result.predictions.filter(p => p.severity === "WARNING").length;
    document.getElementById("inf-normal").textContent = result.normal_records;

    const tbody = document.querySelector("#inference-results-table tbody");
    tbody.innerHTML = "";

    globalState.predictions.slice(0, 40).forEach(pred => {
      const tr = document.createElement("tr");
      const sevClass = pred.severity === "CRITICAL" ? "badge-critical" : (pred.severity === "WARNING" ? "badge-warning" : "badge-normal");
      
      tr.innerHTML = `
        <td>${pred.index}</td>
        <td>${(pred.window || "").slice(0, 19)}</td>
        <td><span class="badge badge-service">${pred.service}</span></td>
        <td><strong>${pred.predicted_category}</strong></td>
        <td><span class="badge ${sevClass}">${pred.severity}</span></td>
        <td>${(pred.confidence * 100).toFixed(1)}%</td>
        <td>
          <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="investigateRecord(${pred.index})">
            🔍 Investigate RCA
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    alert("Inference error: " + err.message);
  }
}

function investigateRecord(index) {
  const pred = globalState.predictions[index];
  if (!pred) return;

  // 1. Fetch SHAP local attribution
  fetchLocalSHAP(index);

  // 2. Run Root Cause Diagnosis
  runRCA(index, pred.service, pred.predicted_category);

  // 3. Switch to RCA tab
  switchTab("tab-rca");
}

// -------------------------------------------------------------
// TAB 5: SHAP EXPLAINABLE AI
// -------------------------------------------------------------
async function fetchGlobalSHAP() {
  try {
    const res = await fetch("/api/explain/global");
    if (!res.ok) return;
    const data = await res.json();
    renderGlobalSHAP(data.global_importances || []);
  } catch (err) {
    console.warn("Global SHAP fetch error:", err);
  }
}

function renderGlobalSHAP(importances) {
  const ctx = document.getElementById("shap-global-chart").getContext("2d");
  if (globalState.charts.shapGlobal) {
    globalState.charts.shapGlobal.destroy();
  }

  const top10 = importances.slice(0, 10);
  const labels = top10.map(i => i.feature.replace("_", " "));
  const values = top10.map(i => i.importance);

  globalState.charts.shapGlobal = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Mean Absolute SHAP Value (Impact on Prediction)",
        data: values,
        backgroundColor: "#06b6d4",
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#94a3b8" } },
        y: { ticks: { color: "#f1f5f9", font: { size: 10 } } }
      },
      plugins: {
        legend: { labels: { color: "#f1f5f9" } }
      }
    }
  });
}

async function fetchLocalSHAP(index) {
  try {
    const res = await fetch(`/api/explain/instance?index=${index}`);
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("shap-selected-label").textContent = data.target_class;
    document.getElementById("shap-narrative-text").textContent = data.natural_language_explanation;

    renderLocalSHAPWaterfall(data.contributions || []);
  } catch (err) {
    console.warn("Local SHAP error:", err);
  }
}

function renderLocalSHAPWaterfall(contributions) {
  const ctx = document.getElementById("shap-local-chart").getContext("2d");
  if (globalState.charts.shapLocal) {
    globalState.charts.shapLocal.destroy();
  }

  const top = contributions.slice(0, 8);
  const labels = top.map(c => `${c.feature} (${c.value})`);
  const values = top.map(c => c.shap_value);
  const colors = top.map(c => c.shap_value > 0 ? "#ef4444" : "#10b981");

  globalState.charts.shapLocal = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "SHAP Value Contribution (+ pushes toward anomaly, - pushes toward normal)",
        data: values,
        backgroundColor: colors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#94a3b8" } },
        y: { ticks: { color: "#f1f5f9", font: { size: 10 } } }
      },
      plugins: {
        legend: { labels: { color: "#f1f5f9" } }
      }
    }
  });
}

// -------------------------------------------------------------
// TAB 6: ROOT CAUSE ANALYSIS & DEPENDENCY TOPOLOGY
// -------------------------------------------------------------
async function initTopologyGraph() {
  const container = document.getElementById("topology-canvas");
  try {
    const res = await fetch("/api/topology");
    if (!res.ok) return;
    const topo = await res.json();
    globalState.topologyData = topo;

    const nodes = new vis.DataSet(
      topo.nodes.map(n => ({
        id: n.id,
        label: n.id,
        shape: n.type === "database" ? "database" : "box",
        color: {
          background: n.type === "database" ? "#065f46" : "#1e3a8a",
          border: n.type === "database" ? "#10b981" : "#3b82f6"
        },
        font: { color: "#ffffff", size: 11 }
      }))
    );

    const edges = new vis.DataSet(
      topo.edges.map(e => ({
        from: e.from,
        to: e.to,
        arrows: "to",
        color: { color: "#334155" },
        smooth: { type: "cubicBezier" }
      }))
    );

    const data = { nodes, edges };
    const options = {
      physics: {
        solver: "forceAtlas2Based",
        forceAtlas2Based: {
          gravitationalConstant: -35,
          centralGravity: 0.01,
          springLength: 100,
          springConstant: 0.08
        },
        maxVelocity: 50,
        minVelocity: 0.1
      },
      interaction: { hover: true, zoomView: true, dragView: true }
    };

    globalState.networkInstance = new vis.Network(container, data, options);

    // Node click to diagnose
    globalState.networkInstance.on("click", params => {
      if (params.nodes.length > 0) {
        const clickedService = params.nodes[0];
        runRCA(0, clickedService, "ANOMALY_INVESTIGATION");
      }
    });

  } catch (err) {
    console.warn("Failed to load topology:", err);
  }
}

async function runRCA(recordIndex, serviceName, category) {
  try {
    const payload = {
      record_index: recordIndex,
      service: serviceName,
      predicted_category: category
    };
    const res = await fetch("/api/rca/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const diag = await res.json();
    if (!res.ok) throw new Error(diag.detail || "RCA failed");

    renderRCA(diag);
  } catch (err) {
    alert("RCA error: " + err.message);
  }
}

function renderRCA(diag) {
  // Update 3-tier badge
  const badge = document.getElementById("rca-tier-badge");
  badge.textContent = diag.diagnosis_tier.replace(/_/g, " ");
  if (diag.diagnosis_tier === "CONFIRMED_ROOT_CAUSE") {
    badge.className = "badge badge-critical";
  } else if (diag.diagnosis_tier === "PROBABLE_ROOT_CAUSE") {
    badge.className = "badge badge-warning";
  } else {
    badge.className = "badge badge-tier";
  }

  document.getElementById("rca-affected-srv").textContent = diag.affected_microservice;
  document.getElementById("rca-root-srv").textContent = diag.root_cause_service;
  document.getElementById("rca-confidence").textContent = `${(diag.confidence_score * 100).toFixed(0)}%`;
  document.getElementById("rca-probable-cause").textContent = diag.probable_root_cause;
  document.getElementById("rca-why-happened").textContent = diag.why_it_happened;
  document.getElementById("rca-evidence-sufficiency").textContent = diag.evidence_sufficiency;

  // Render Blast Radius
  const upList = document.getElementById("rca-upstream-list");
  upList.innerHTML = "";
  const upstream = diag.blast_radius?.upstream_callers_affected || [];
  if (upstream.length === 0) {
    upList.innerHTML = "<span style='color: var(--text-muted); font-size: 11px;'>No upstream callers affected.</span>";
  } else {
    upstream.forEach(u => {
      const sp = document.createElement("span");
      sp.className = "badge badge-tier";
      sp.textContent = u;
      upList.appendChild(sp);
    });
  }

  const downList = document.getElementById("rca-downstream-list");
  downList.innerHTML = "";
  const downstream = diag.blast_radius?.downstream_dependencies || [];
  if (downstream.length === 0) {
    downList.innerHTML = "<span style='color: var(--text-muted); font-size: 11px;'>No downstream dependencies.</span>";
  } else {
    downstream.forEach(d => {
      const sp = document.createElement("span");
      sp.className = "badge badge-service";
      sp.textContent = d;
      downList.appendChild(sp);
    });
  }

  // Highlight in Topology Canvas
  if (globalState.networkInstance && globalState.topologyData) {
    const nodesDataSet = globalState.networkInstance.body.data.nodes;
    const updates = [];

    globalState.topologyData.nodes.forEach(n => {
      const isRoot = n.id === diag.root_cause_service;
      const isAffected = n.id === diag.affected_microservice;
      const isUpstream = upstream.includes(n.id);

      let bgColor = n.type === "database" ? "#065f46" : "#1e3a8a";
      let borderColor = n.type === "database" ? "#10b981" : "#3b82f6";

      if (isRoot) {
        bgColor = "#7f1d1d";
        borderColor = "#ef4444";
      } else if (isAffected) {
        bgColor = "#78350f";
        borderColor = "#f59e0b";
      } else if (isUpstream) {
        bgColor = "#581c87";
        borderColor = "#8b5cf6";
      }

      updates.push({ id: n.id, color: { background: bgColor, border: borderColor } });
    });

    nodesDataSet.update(updates);
    globalState.networkInstance.focus(diag.root_cause_service, {
      scale: 1.2,
      animation: { duration: 600 }
    });
  }

  // Render Multi-source Supporting Evidence
  const logsDiv = document.getElementById("rca-evidence-logs");
  const tracesDiv = document.getElementById("rca-evidence-traces");
  const eventsDiv = document.getElementById("rca-evidence-events");

  const logs = diag.supporting_evidence?.relevant_logs || [];
  if (logs.length === 0) {
    logsDiv.textContent = "No corroborating error logs found.";
  } else {
    logsDiv.innerHTML = logs.map(l => 
      `<div style="margin-bottom: 4px;"><span style="color: #ef4444;">[${l.level}]</span> <strong>${l.service}</strong>: ${l.message} ${l.exception ? `<br><span style="color: #f87171;">Exception: ${l.exception}</span>` : ''}</div>`
    ).join("");
  }

  const traces = diag.supporting_evidence?.relevant_traces || [];
  if (traces.length === 0) {
    tracesDiv.textContent = "No slow or failing trace spans recorded.";
  } else {
    tracesDiv.innerHTML = traces.map(t => 
      `<div>Trace <code>${t.trace_id}</code>: <strong>${t.service}</strong> -> <code>${t.operation}</code> took <strong>${t.duration}ms</strong> ${t.has_error ? '<span style="color: #ef4444;">(FAILED)</span>' : ''}</div>`
    ).join("");
  }

  const events = diag.supporting_evidence?.relevant_events || [];
  if (events.length === 0) {
    eventsDiv.textContent = "No pod crash or restart events recorded.";
  } else {
    eventsDiv.innerHTML = events.map(e => 
      `<div><span style="color: #ef4444;">[${e.event_type}]</span> <strong>${e.service}</strong>: ${e.message}</div>`
    ).join("");
  }

  // Populate Tab 7 Recommendations Playbook
  renderPlaybook(diag);
}

function renderPlaybook(diag) {
  document.getElementById("playbook-title").textContent = diag.playbook_title || "Remediation Playbook";
  document.getElementById("playbook-severity").textContent = diag.diagnosis_tier === "CONFIRMED_ROOT_CAUSE" ? "CRITICAL" : "HIGH";

  const stepsContainer = document.getElementById("playbook-steps");
  stepsContainer.innerHTML = "";

  const steps = diag.recommended_troubleshooting_steps || [];
  steps.forEach(step => {
    const div = document.createElement("div");
    div.style.cssText = "padding: 12px 16px; background-color: var(--bg-secondary); border-radius: 8px; border-left: 3px solid var(--accent-cyan); font-size: 13px;";
    div.textContent = step;
    stepsContainer.appendChild(div);
  });
}
