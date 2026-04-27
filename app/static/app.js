'use strict';

const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const fileNameEl   = document.getElementById('file-name');
const colMap       = document.getElementById('col-map');
const selScore     = document.getElementById('sel-score');
const selA1        = document.getElementById('sel-a1');
const selA2        = document.getElementById('sel-a2');
const thresholdEl  = document.getElementById('threshold');
const thresholdVal = document.getElementById('threshold-value');
const runBtn       = document.getElementById('run-btn');
const progressSect = document.getElementById('progress-section');
const barFill      = document.getElementById('progress-fill');
const msgEl        = document.getElementById('progress-message');
const pctEl        = document.getElementById('progress-percent');
const cancelBtn    = document.getElementById('cancel-btn');
const downloadBtn  = document.getElementById('download-btn');
const progressStatus = document.getElementById('progress-status');
const historyBody  = document.getElementById('history-body');
const historyEmpty = document.getElementById('history-empty');
const historyTable = document.getElementById('history-table');

let uploadId = null;
let originalFilename = 'upload.xlsx';
let currentRunId = null;

// ── Threshold ─────────────────────────────────────────────
thresholdEl.addEventListener('input', () => {
  thresholdVal.textContent = parseFloat(thresholdEl.value).toFixed(2);
});

// ── Drag & Drop ───────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

async function handleFile(f) {
  fileNameEl.textContent = f.name;
  fileNameEl.classList.add('visible');
  runBtn.disabled = true;
  colMap.classList.add('hidden');

  const form = new FormData();
  form.append('file', f);

  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: form });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    uploadId = data.upload_id;
    originalFilename = data.filename;

    populateSelects(data.columns, data.guessed);
    colMap.classList.remove('hidden');
    runBtn.disabled = false;
  } catch (err) {
    alert('Ошибка загрузки файла: ' + err.message);
  }
}

function populateSelects(columns, guessed) {
  [selScore, selA1, selA2].forEach(sel => {
    sel.innerHTML = '';
    columns.forEach(col => {
      const opt = document.createElement('option');
      opt.value = col;
      opt.textContent = col;
      sel.appendChild(opt);
    });
  });
  if (guessed.score_col) selScore.value = guessed.score_col;
  if (guessed.a1_col)    selA1.value    = guessed.a1_col;
  if (guessed.a2_col)    selA2.value    = guessed.a2_col;
}

function resetUploadState() {
  uploadId = null;
  fileInput.value = '';          // allow re-selecting the same file
  fileNameEl.textContent = '';
  fileNameEl.classList.remove('visible');
  colMap.classList.add('hidden');
  runBtn.disabled = true;
}

// ── Run ───────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
  if (!uploadId) {
    alert('Выберите Excel-файл перед запуском.');
    return;
  }

  runBtn.disabled = true;
  downloadBtn.classList.add('hidden');
  cancelBtn.classList.remove('hidden');
  progressSect.classList.remove('hidden');
  setProgress(0, 'Отправка задачи...');

  const form = new FormData();
  form.append('upload_id', uploadId);
  form.append('score_col', selScore.value);
  form.append('a1_col', selA1.value);
  form.append('a2_col', selA2.value);
  form.append('threshold', thresholdEl.value);
  form.append('original_filename', originalFilename);

  let runId;
  try {
    const resp = await fetch('/api/run', { method: 'POST', body: form });
    if (!resp.ok) throw new Error(await resp.text());
    ({ run_id: runId } = await resp.json());
  } catch (err) {
    setProgress(0, 'Ошибка запуска: ' + err.message, 'failed');
    runBtn.disabled = false;
    cancelBtn.classList.add('hidden');
    return;
  }

  currentRunId = runId;
  uploadId = null; // consumed

  const es = new EventSource(`/api/progress/${runId}`);
  es.onmessage = (e) => {
    const state = JSON.parse(e.data);
    setProgress(state.percent, state.message, state.status);

    if (state.status === 'completed') {
      es.close();
      resetUploadState();
      cancelBtn.classList.add('hidden');
      downloadBtn.href = `/api/runs/${runId}/download`;
      downloadBtn.classList.remove('hidden');
      runBtn.disabled = false;
      loadHistory();
    } else if (state.status === 'failed' || state.status === 'cancelled') {
      es.close();
      resetUploadState();
      cancelBtn.classList.add('hidden');
      runBtn.disabled = false;
      loadHistory();
    }
  };
  es.onerror = () => es.close();
});

// ── Cancel ────────────────────────────────────────────────
cancelBtn.addEventListener('click', async () => {
  if (!currentRunId) return;
  cancelBtn.disabled = true;
  cancelBtn.textContent = 'Отмена...';
  try {
    await fetch(`/api/runs/${currentRunId}/cancel`, { method: 'POST' });
  } finally {
    cancelBtn.disabled = false;
    cancelBtn.textContent = 'Отменить';
  }
});

function setProgress(pct, msg, status) {
  barFill.style.width = pct + '%';
  msgEl.textContent = msg;
  pctEl.textContent = pct + '%';
  msgEl.style.color = status === 'failed' ? 'var(--red)' : '';

  const statusBadge = { failed: ['failed', 'Ошибка'], cancelled: ['cancelled', 'Отменено'] };
  if (statusBadge[status]) {
    const [cls, label] = statusBadge[status];
    progressStatus.className = `badge ${cls}`;
    progressStatus.textContent = label;
    progressStatus.classList.remove('hidden');
  } else {
    progressStatus.className = 'badge hidden';
    progressStatus.textContent = '';
  }
}

// ── History ───────────────────────────────────────────────
async function loadHistory() {
  const runs = await fetch('/api/runs').then(r => r.json()).catch(() => []);

  if (!runs.length) {
    historyEmpty.style.display = 'block';
    historyTable.style.display = 'none';
    return;
  }

  historyEmpty.style.display = 'none';
  historyTable.style.display = 'table';
  historyBody.innerHTML = '';

  runs.forEach(run => {
    const tr = document.createElement('tr');

    const dt = new Date(run.timestamp);
    const dateStr =
      dt.toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit', year: '2-digit' }) +
      ' ' +
      dt.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });

    const statusMap = {
      completed: ['completed', 'Готово'],
      failed:    ['failed',    'Ошибка'],
      cancelled: ['cancelled', 'Отменено'],
      running:   ['running',   'В процессе'],
    };
    const [cls, label] = statusMap[run.status] || ['running', run.status];

    const dlCell = run.status === 'completed'
      ? `<a class="dl-link" href="/api/runs/${run.id}/download">
           <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
             <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
             <polyline points="7 10 12 15 17 10"/>
             <line x1="12" y1="15" x2="12" y2="3"/>
           </svg>Скачать</a>`
      : '—';

    tr.innerHTML = `
      <td>${dateStr}</td>
      <td class="filename-cell" title="${escHtml(run.input_filename)}">${escHtml(run.input_filename)}</td>
      <td>${Number(run.threshold).toFixed(2)}</td>
      <td>${run.row_count || '—'}</td>
      <td><span class="badge ${cls}">${label}</span></td>
      <td>${dlCell}</td>
    `;
    historyBody.appendChild(tr);
  });
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

loadHistory();
