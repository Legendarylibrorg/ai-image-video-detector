const qs = (s) => document.querySelector(s);
const qsa = (s) => Array.from(document.querySelectorAll(s));

const tabs = qsa('.tab');
const panels = qsa('.tab-panel');
for (const t of tabs) {
  t.addEventListener('click', () => {
    const tab = t.dataset.tab;
    tabs.forEach((x) => x.classList.toggle('active', x === t));
    panels.forEach((p) => p.classList.toggle('active', p.dataset.panel === tab));
  });
}

const statusEl = qs('#status');
const setStatus = (m) => { if (statusEl) statusEl.textContent = m; };
const clamp = (v) => Math.max(0, Math.min(1, Number(v || 0)));

const renderJson = (id, data) => {
  const el = qs(id);
  if (!el) return;
  el.textContent = JSON.stringify(data, null, 2);
};

const postJson = async (url, body) => {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

const postFile = async (url, file) => {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(url, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

// Image detect
let selectedFile = null;
const fileInput = qs('#fileInput');
const dropzone = qs('#dropzone');
const analyzeBtn = qs('#analyzeBtn');
const preview = qs('#preview');

const metricsNode = (title, value) => {
  const d = document.createElement('article');
  d.className = 'metric';
  d.innerHTML = `<h4>${title}</h4><p>${value}</p>`;
  return d;
};

const renderFlags = (id, flags) => {
  const ul = qs(id);
  if (!ul) return;
  ul.innerHTML = '';
  const list = Array.isArray(flags) && flags.length ? flags : ['none'];
  for (const f of list) {
    const li = document.createElement('li');
    li.textContent = String(f);
    ul.appendChild(li);
  }
};

const setBadge = (label) => {
  const badge = qs('#resultBadge');
  if (!badge) return;
  badge.textContent = String(label || '-').toUpperCase();
  badge.style.borderColor = label === 'AI-generated' ? '#f26f7f' : label === 'Unknown' ? '#f2be68' : '#2dd8a3';
};

const handleFile = (f) => {
  selectedFile = f;
  analyzeBtn.disabled = false;
  setStatus(`${f.name} ready`);
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = String(reader.result);
    preview.style.display = 'block';
  };
  reader.readAsDataURL(f);
};

fileInput?.addEventListener('change', (e) => {
  const f = e.target.files?.[0];
  if (f) handleFile(f);
});

['dragenter', 'dragover'].forEach((ev) => {
  dropzone?.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add('drag');
  });
});
['dragleave', 'drop'].forEach((ev) => {
  dropzone?.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag');
  });
});
dropzone?.addEventListener('drop', (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) handleFile(f);
});

analyzeBtn?.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  setStatus('Analyzing image...');
  try {
    const form = new FormData();
    form.append('image', selectedFile);
    const res = await fetch('/detect', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();

    qs('#resultLabel').textContent = `${d.label || 'Result'} (${d.domain || 'n/a'})`;
    setBadge(d.label);

    const metrics = qs('#imageMetrics');
    metrics.innerHTML = '';
    metrics.appendChild(metricsNode('AI Probability', clamp(d.prob_ai).toFixed(3)));
    metrics.appendChild(metricsNode('Combined Risk', clamp(d.combined_risk).toFixed(3)));
    metrics.appendChild(metricsNode('Metadata', clamp(d.metadata_score).toFixed(3)));
    metrics.appendChild(metricsNode('OOD', clamp(d.ood_score).toFixed(3)));
    metrics.appendChild(metricsNode('Text', clamp(d.text_score).toFixed(3)));

    renderFlags('#metadataFlags', d.metadata_flags);
    renderFlags('#provenanceFlags', d.provenance_flags);
    renderFlags('#oodFlags', d.ood_flags);
    renderFlags('#textFlags', d.text_flags);
    renderJson('#imageJson', d);

    qs('#resultPanel').classList.remove('hidden');
    setStatus('Image analysis complete');
  } catch (e) {
    setStatus(`Error: ${e.message || e}`);
  } finally {
    analyzeBtn.disabled = false;
  }
});

// Text/conversation/url
qs('#textBtn')?.addEventListener('click', async () => {
  try { renderJson('#textOut', await postJson('/analyze/text', { text: qs('#textInput').value })); }
  catch (e) { renderJson('#textOut', { error: String(e) }); }
});

qs('#conversationBtn')?.addEventListener('click', async () => {
  try { renderJson('#conversationOut', await postJson('/analyze/conversation', { text: qs('#conversationInput').value })); }
  catch (e) { renderJson('#conversationOut', { error: String(e) }); }
});

qs('#urlBtn')?.addEventListener('click', async () => {
  try { renderJson('#urlOut', await postJson('/analyze/url', { url: qs('#urlInput').value })); }
  catch (e) { renderJson('#urlOut', { error: String(e) }); }
});

// PDF/audio
qs('#pdfBtn')?.addEventListener('click', async () => {
  const f = qs('#pdfInput').files?.[0];
  if (!f) return renderJson('#pdfOut', { error: 'Choose a PDF file first' });
  try { renderJson('#pdfOut', await postFile('/analyze/pdf', f)); }
  catch (e) { renderJson('#pdfOut', { error: String(e) }); }
});

qs('#audioBtn')?.addEventListener('click', async () => {
  const f = qs('#audioInput').files?.[0];
  if (!f) return renderJson('#audioOut', { error: 'Choose an audio file first' });
  try { renderJson('#audioOut', await postFile('/analyze/audio', f)); }
  catch (e) { renderJson('#audioOut', { error: String(e) }); }
});

// Fusion
qs('#fusionBtn')?.addEventListener('click', async () => {
  const fields = {
    image: qs('#fImage').value,
    video: qs('#fVideo').value,
    text: qs('#fText').value,
    conversation: qs('#fConversation').value,
    url: qs('#fUrl').value,
    pdf: qs('#fPdf').value,
    audio: qs('#fAudio').value,
  };
  const scores = {};
  for (const [k, v] of Object.entries(fields)) {
    if (v !== '' && !Number.isNaN(Number(v))) scores[k] = clamp(v);
  }
  try { renderJson('#fusionOut', await postJson('/analyze/multimodal', { scores })); }
  catch (e) { renderJson('#fusionOut', { error: String(e) }); }
});
