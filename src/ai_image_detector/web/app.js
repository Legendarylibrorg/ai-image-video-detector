const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const previewEl = document.getElementById("preview");
const resultPanel = document.getElementById("resultPanel");

let selectedFile = null;

const setStatus = (msg) => {
  statusEl.textContent = msg;
};

const clamp01 = (v) => Math.max(0, Math.min(1, Number(v || 0)));

const fillMeter = (id, value) => {
  const el = document.getElementById(id);
  el.style.width = `${Math.round(clamp01(value) * 100)}%`;
};

const renderFlags = (id, flags) => {
  const ul = document.getElementById(id);
  ul.innerHTML = "";
  const list = Array.isArray(flags) && flags.length ? flags : ["none"];
  for (const f of list) {
    const li = document.createElement("li");
    li.textContent = String(f);
    ul.appendChild(li);
  }
};

const setBadge = (label) => {
  const badge = document.getElementById("resultBadge");
  badge.textContent = String(label || "-").toUpperCase();
  badge.style.borderColor = label === "AI-generated" ? "#ff7089" : label === "Unknown" ? "#ffc46b" : "#47d8a5";
};

const showPreview = (file) => {
  const reader = new FileReader();
  reader.onload = () => {
    previewEl.src = String(reader.result);
    previewEl.style.display = "block";
  };
  reader.readAsDataURL(file);
};

const handleFile = (file) => {
  selectedFile = file;
  analyzeBtn.disabled = false;
  showPreview(file);
  setStatus(`${file.name} ready`);
};

fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) handleFile(file);
});

["dragenter", "dragover"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("drag");
  });
});

["dragleave", "drop"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("drag");
  });
});

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  setStatus("Analyzing image...");

  try {
    const form = new FormData();
    form.append("image", selectedFile);

    const res = await fetch("/detect", { method: "POST", body: form });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }

    const data = await res.json();

    document.getElementById("resultLabel").textContent = data.label || "Result";
    setBadge(data.label);

    document.getElementById("probAi").textContent = clamp01(data.prob_ai).toFixed(3);
    document.getElementById("combinedRisk").textContent = clamp01(data.combined_risk).toFixed(3);
    document.getElementById("metadataScore").textContent = clamp01(data.metadata_score).toFixed(3);
    document.getElementById("oodScore").textContent = clamp01(data.ood_score).toFixed(3);

    fillMeter("probBar", data.prob_ai);
    fillMeter("riskBar", data.combined_risk);
    fillMeter("metaBar", data.metadata_score);
    fillMeter("oodBar", data.ood_score);

    renderFlags("metadataFlags", data.metadata_flags);
    renderFlags("provenanceFlags", data.provenance_flags);
    renderFlags("oodFlags", data.ood_flags);

    document.getElementById("thresholdVal").textContent = Number(data.threshold || 0).toFixed(3);
    document.getElementById("unknownVal").textContent = Number(data.unknown_margin || 0).toFixed(3);
    document.getElementById("modelCountVal").textContent = String(data.model_count ?? "-");
    document.getElementById("versionVal").textContent = String(data.model_version ?? "-");

    resultPanel.classList.remove("hidden");
    setStatus("Analysis complete");
  } catch (err) {
    setStatus(`Error: ${err.message || err}`);
  } finally {
    analyzeBtn.disabled = false;
  }
});
