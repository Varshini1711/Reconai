// ReconAI dashboard frontend - vanilla JS, no build step.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ---------------- Tabs ----------------
$$(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    $$(".tab-btn").forEach(b => b.classList.remove("active"));
    $$(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ---------------- Dashboard ----------------
async function loadSummary() {
  const res = await fetch("/api/summary");
  const data = await res.json();

  $("#datasetBadge").textContent = `${data.dataset} dataset`;
  $("#cardTotal").textContent = data.total_records;
  $("#cardAuto").textContent = data.auto_matched;
  $("#cardReview").textContent = data.needs_review;
  $("#cardUnresolved").textContent = data.unresolved;
  $("#matchRateBar").style.width = `${data.match_rate}%`;
  $("#matchRateText").textContent = `${data.match_rate}% auto-matched without human review`;

  $("#llmStatus").textContent = data.llm_key_configured
    ? "✓ Real LLM agent active for Layer 5 (Gemini)."
    : "⚠ No LLM API key configured — Layer 5 cases will show NEEDS_REVIEW ('AI investigation unavailable') instead of real reasoning.";

  if (data.evaluation && !data.evaluation.error) {
    $("#evalPanel").style.display = "block";
    const e = data.evaluation;
    $("#mPrecision").textContent = e.precision != null ? `${e.precision.toFixed(1)}%` : "n/a";
    $("#mCoverage").textContent = e.coverage != null ? `${e.coverage.toFixed(1)}%` : "n/a";
    $("#mEscalation").textContent = e.escalation_correctness != null ? `${e.escalation_correctness.toFixed(1)}%` : "n/a";
    $("#mUnresolved").textContent = e.unresolved_accuracy != null ? `${e.unresolved_accuracy.toFixed(1)}%` : "n/a";
    $("#mFalseMatch").textContent = e.false_match_rate != null ? `${e.false_match_rate.toFixed(1)}%` : "n/a";
  } else {
    $("#evalPanel").style.display = "none";
  }
}

// ---------------- Results table ----------------
let currentFilter = "ALL";

async function loadResults() {
  const res = await fetch(`/api/records?status=${currentFilter}`);
  const rows = await res.json();
  const body = $("#resultsBody");
  body.innerHTML = "";

  if (rows.length === 0) {
    body.innerHTML = `<tr><td colspan="5" class="muted">No records match this filter.</td></tr>`;
    return;
  }

  rows.forEach(r => {
    const tr = document.createElement("tr");
    tr.addEventListener("click", () => openDetail(r.case_ref));
    const matched = [...r.settlement_ids].join(", ") || "—";
    const conf = r.confidence != null ? `${Math.round(r.confidence)}%` : "—";
    tr.innerHTML = `
      <td>${r.transaction_ids.join(", ") || r.case_ref}</td>
      <td>${matched}</td>
      <td><span class="badge badge-${r.decision}">${r.decision.replace("_", " ")}</span></td>
      <td>${conf}</td>
      <td>${formatMethod(r.match_method)}</td>
    `;
    body.appendChild(tr);
  });
}

function formatMethod(method) {
  const map = {
    layer1_exact: "Layer 1 — Exact Match",
    layer3_fuzzy: "Layer 3 — Fuzzy Match",
    layer4_many_to_one: "Layer 4 — Grouped (Many→One)",
    layer4_one_to_many: "Layer 4 — Grouped (One→Many)",
    layer4_refund_adjusted: "Layer 4 — Refund-Adjusted",
    layer4_fee_adjusted: "Layer 4 — Fee-Adjusted",
    layer4_partial_settlement: "Layer 4 — Partial Settlement",
    layer5_agent: "Layer 5 — AI Investigation",
    orphan_settlement: "Orphan Settlement",
    orphan_refund: "Orphan Refund",
  };
  return map[method] || method || "—";
}

$$(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    $$(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.status;
    loadResults();
  });
});

// ---------------- Record detail modal ----------------
async function openDetail(caseRef) {
  const res = await fetch(`/api/records/${caseRef}`);
  if (!res.ok) return;
  const d = await res.json();

  const txnRows = d.transactions.map(t =>
    `<div class="record-row"><span>${t.txn_id}</span><span>₹${t.amount} · ${t.merchant_name} · ${t.date}</span></div>`
  ).join("") || `<p class="muted small">None</p>`;

  const setRows = d.settlements.map(s =>
    `<div class="record-row"><span>${s.settlement_id}</span><span>₹${s.amount} · ${s.merchant_name} · ${s.date}</span></div>`
  ).join("") || `<p class="muted small">None</p>`;

  const refundRows = d.refunds.map(r =>
    `<div class="record-row"><span>${r.refund_id}</span><span>₹${r.amount} on ${r.date}</span></div>`
  ).join("") || `<p class="muted small">None</p>`;

  const feeRows = d.fees.map(f =>
    `<div class="record-row"><span>${f.fee_id}</span><span>₹${f.amount} (${f.type})</span></div>`
  ).join("") || `<p class="muted small">None</p>`;

  const evidenceHtml = (d.evidence || []).map(e => `<div class="evidence-item">${escapeHtml(e)}</div>`).join("");

  let aiPanel = "";
  if (d.is_ai_investigated) {
    const auditHtml = (d.audit_trail || [])
      .filter(s => s.startsWith("Layer 5"))
      .map(s => `<div class="audit-step">${escapeHtml(s)}</div>`).join("");
    aiPanel = `
      <div class="detail-section">
        <h3>AI Investigation (Layer 5)</h3>
        <span class="ai-flag">Real LLM agent — tool calls, evidence, self-critique below</span>
        <div style="margin-top:10px;">${auditHtml || '<p class="muted small">No tool trace available.</p>'}</div>
      </div>`;
  }

  const badgeClass = `badge-${d.decision}`;
  const decisionColor = d.decision === "AUTO_MATCHED" ? "var(--green)" : d.decision === "NEEDS_REVIEW" ? "var(--amber)" : "var(--red)";

  $("#modalContent").innerHTML = `
    <h2>${d.case_ref}</h2>
    <div class="detail-section">
      <h3>Transactions</h3>${txnRows}
    </div>
    <div class="detail-section">
      <h3>Settlements</h3>${setRows}
    </div>
    <div class="detail-section">
      <h3>Refunds</h3>${refundRows}
    </div>
    <div class="detail-section">
      <h3>Fees</h3>${feeRows}
    </div>
    <div class="detail-section">
      <h3>Why did ReconAI make this decision?</h3>
      ${evidenceHtml || '<p class="muted small">No evidence recorded.</p>'}
      <p style="margin-top:10px; font-size:13px;">${escapeHtml(d.reasoning || "")}</p>
    </div>
    ${aiPanel}
    <div class="decision-final" style="background:${decisionColor}22; color:${decisionColor};">
      ${d.decision.replace("_", " ")} — Confidence: ${d.confidence != null ? Math.round(d.confidence) : "—"}%
    </div>
  `;
  $("#detailModal").classList.add("open");
}

$("#closeModal").addEventListener("click", () => $("#detailModal").classList.remove("open"));
$("#detailModal").addEventListener("click", (e) => { if (e.target.id === "detailModal") $("#detailModal").classList.remove("open"); });

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------- Ask the Ledger ----------------
$("#chatForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#chatInput");
  const question = input.value.trim();
  if (!question) return;

  appendChat(question, "chat-user");
  input.value = "";

  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await res.json();
  appendChat(data.answer, "chat-assistant");
});

function appendChat(text, cls) {
  const log = $("#chatLog");
  const div = document.createElement("div");
  div.className = `chat-msg ${cls}`;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

// ---------------- Run / Upload ----------------
let runInProgress = false;

async function runPipeline(useUploaded) {
  if (runInProgress) return;   // guard against double-clicks firing overlapping runs / wasting API quota
  runInProgress = true;

  const demoBtn = $("#runDemoBtn");
  const uploadBtn = $("#runUploadedBtn");
  demoBtn.disabled = true;
  uploadBtn.disabled = true;

  $("#runProgress").style.display = "block";
  const heading = $("#runProgress h2");
  const list = $("#progressList");
  const layers = [
    "Layer 1 — Exact matching",
    "Layer 2 — Candidate generation",
    "Layer 3 — Fuzzy matching",
    "Layer 4 — Grouped reconciliation",
    "Layer 5 — AI investigation (can take several minutes if many cases need real AI review)",
  ];
  list.innerHTML = layers.map(l => `<li>${l}</li>`).join("");
  heading.textContent = "Running... (do not close this tab)";

  // Honest elapsed-time counter instead of a fake progress animation -
  // we genuinely don't know how far along the single blocking backend
  // call is, so we say so rather than implying false completeness.
  const startTime = Date.now();
  const elapsedTimer = setInterval(() => {
    const secs = Math.floor((Date.now() - startTime) / 1000);
    heading.textContent = `Running... (${secs}s elapsed — do not close this tab)`;
  }, 1000);

  try {
    const res = await fetch(`/api/run?use_uploaded=${useUploaded}`, { method: "POST" });
    const data = await res.json();
    clearInterval(elapsedTimer);

    if (!res.ok) {
      heading.textContent = "Run failed.";
      alert(data.detail || "Run failed.");
      return;
    }

    $$("#progressList li").forEach(it => it.classList.add("done"));
    heading.textContent = "Done.";

    await loadSummary();
    await loadResults();
    $$(".tab-btn").forEach(b => b.classList.remove("active"));
    $$(".tab-panel").forEach(p => p.classList.remove("active"));
    $('[data-tab="dashboard"]').classList.add("active");
    $("#tab-dashboard").classList.add("active");
    $("#runProgress").style.display = "none";   // was never reset before - real bug fix
  } catch (err) {
    clearInterval(elapsedTimer);
    heading.textContent = "Run failed.";
    alert("Run failed: " + err);
  } finally {
    runInProgress = false;
    demoBtn.disabled = false;
    uploadBtn.disabled = false;
  }
}

$("#runDemoBtn").addEventListener("click", () => runPipeline(false));
$("#runUploadedBtn").addEventListener("click", () => runPipeline(true));

$("#uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const res = await fetch("/api/upload", { method: "POST", body: formData });
  const data = await res.json();
  if (!res.ok) {
    $("#uploadStatus").textContent = "Error: " + (Array.isArray(data.detail) ? data.detail.join("; ") : data.detail);
    $("#uploadStatus").style.color = "var(--red)";
    return;
  }
  $("#uploadStatus").textContent = data.message;
  $("#uploadStatus").style.color = "var(--green)";
  $("#runUploadedBtn").disabled = false;
});

// ---------------- Init ----------------
loadSummary();
loadResults();
