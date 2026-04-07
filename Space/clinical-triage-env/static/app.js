/* app.js — ClinicalTriageEnv Dashboard */

'use strict';

// ── State ────────────────────────────────────────────────────────────────────

const STATE = {
  obs:            null,     // current observation from server
  selectedId:     null,     // patient_id selected in manual mode
  mode:           'manual', // 'manual' | 'ai'
  running:        false,
  aiTimer:        null,
  stepScores:     [],       // history for sparkline
  totalScore:     0,
  stepCount:      0,
  taskId:         'triage-basics', // Default task
};

const MAX_LOG   = 80;
const MAX_CHART = 40;

// ── DOM refs ─────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

const ui = {
  simClock:     $('simClock'),
  stepCount:    $('stepCount'),
  diagPill:     $('diagPill'),
  diagLabel:    $('diagLabel'),
  diagModal:    $('diagModal'),
  diagContent:  $('diagContent'),
  seedInput:    $('seedInput'),
  randomSeed:   $('randomSeed'),
  modeManual:   $('modeManual'),
  modeAI:       $('modeAI'),
  speedGroup:   $('speedGroup'),
  speedSlider:  $('speedSlider'),
  speedLabel:   $('speedLabel'),
  startBtn:     $('startBtn'),
  stopBtn:      $('stopBtn'),
  patientGrid:  $('patientGrid'),
  queueMeta:    $('queueMeta'),
  actionPanel:  $('actionPanel'),
  selInfo:      $('selectedPatientInfo'),
  actionBtns:   document.querySelectorAll('.action-btn'),
  esiBtns:      document.querySelectorAll('.esi-btn'),
  bigScore:     $('bigScore'),
  scoreSub:     $('scoreSub'),
  scoreChart:   $('scoreChart'),
  lastScoreBox: $('lastScoreBox'),
  lastScoreVal: $('lastScoreVal'),
  lastScoreBar: $('lastScoreBar'),
  lastRationale:$('lastRationale'),
  episodeLog:   $('episodeLog'),
  bedBar:$('bedBar'), bedNum:$('bedNum'),
  labBar:$('labBar'), labNum:$('labNum'),
  imgBar:$('imgBar'), imgNum:$('imgNum'),
  tokBar:$('tokBar'), tokNum:$('tokNum'),
};

// ── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  checkDiagnostics();
  attachEvents();
  setInterval(checkDiagnostics, 60_000);
});

function attachEvents() {
  ui.startBtn.addEventListener('click', startEpisode);
  ui.stopBtn.addEventListener('click',  stopAI);
  ui.randomSeed.addEventListener('click', () => {
    ui.seedInput.value = Math.floor(Math.random() * 9999);
  });

  ui.modeManual.addEventListener('click', () => setMode('manual'));
  ui.modeAI.addEventListener('click',     () => setMode('ai'));

  ui.speedSlider.addEventListener('input', () => {
    ui.speedLabel.textContent = (ui.speedSlider.value / 1000).toFixed(1) + 's';
  });

  // Action buttons
  document.querySelectorAll('.action-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (STATE.mode !== 'manual') return;
      submitManualAction(btn.dataset.action, null);
    });
  });

  // ESI buttons
  document.querySelectorAll('.esi-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (STATE.mode !== 'manual') return;
      submitManualAction('assign_triage', parseInt(btn.dataset.esi));
    });
  });

  // Diagnostics
  ui.diagPill.addEventListener('click', openDiagnostics);
  $('closeDiag').addEventListener('click', () => { ui.diagModal.style.display = 'none'; });

  $('clearLog').addEventListener('click', () => {
    ui.episodeLog.innerHTML = '<div class="log-empty">Log cleared</div>';
  });
}

// ── Mode ──────────────────────────────────────────────────────────────────────

function setMode(mode) {
  STATE.mode = mode;
  ui.modeManual.classList.toggle('active', mode === 'manual');
  ui.modeAI.classList.toggle('active',     mode === 'ai');
  ui.speedGroup.style.display = mode === 'ai' ? 'block' : 'none';
  ui.actionPanel.classList.toggle('ai-mode', mode === 'ai');
  if (mode !== 'ai') stopAI();
}

// ── Episode start ─────────────────────────────────────────────────────────────

async function startEpisode() {
  stopAI();
  STATE.obs        = null;
  STATE.selectedId = null;
  STATE.stepScores = [];
  STATE.totalScore = 0;
  STATE.stepCount  = 0;

  ui.bigScore.textContent  = '—';
  ui.lastScoreBox.style.display = 'none';
  ui.episodeLog.innerHTML  = '<div class="log-empty">Episode starting…</div>';
  ui.patientGrid.innerHTML = '<div class="empty-state"><div class="empty-icon">⟳</div><div>Loading patients…</div></div>';
  ui.startBtn.disabled = true;

  const seed = parseInt(ui.seedInput.value) || 42;

  try {
    // OpenEnv V1 uses /reset
    const res  = await fetch(`/reset`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: STATE.taskId, seed: seed })
    });
    const data = await res.json();
    const obs = data.observation;

    STATE.obs = obs;
    renderObs(obs);
    ui.episodeLog.innerHTML = '';
    appendLog(0, '—', 'Episode started', null, `${obs.patients.length} patients in queue. Seed: ${seed}`);
    ui.startBtn.disabled = false;
    ui.stopBtn.disabled  = false;

    if (STATE.mode === 'ai') startAILoop();
  } catch (e) {
    ui.patientGrid.innerHTML = `<div class="empty-state"><div>Error: ${e.message}</div></div>`;
    ui.startBtn.disabled = false;
  }
}

// ── AI loop ───────────────────────────────────────────────────────────────────

function startAILoop() {
  if (!STATE.obs || STATE.obs.episode_done) return;
  STATE.running = true;
  scheduleAIStep();
}

function scheduleAIStep() {
  const delay = parseInt(ui.speedSlider.value);
  STATE.aiTimer = setTimeout(runAIStep, delay);
}

async function runAIStep() {
  if (!STATE.running || !STATE.obs || STATE.obs.episode_done) {
    stopAI(); return;
  }

  try {
    // 1. Get action from baseline
    const baseRes = await fetch('/baseline', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: STATE.taskId, observation: STATE.obs }),
    });
    const baseData = await baseRes.json();

    // 2. Submit to /step (OpenEnv standard)
    await submitAction(baseData.action);

    if (!STATE.obs.episode_done && STATE.running) scheduleAIStep();
    else stopAI();
  } catch (e) {
    appendLog(STATE.stepCount, '—', 'AI error: ' + e.message, null, '');
    stopAI();
  }
}

// ── Manual action ─────────────────────────────────────────────────────────────

async function submitManualAction(actionType, esiLevel) {
  if (!STATE.obs || !STATE.selectedId) return;

  const action = {
    patient_id:  STATE.selectedId,
    action_type: actionType,
    esi_level:   esiLevel || null,
  };

  // Disable buttons during request
  setActionButtonsEnabled(false);
  await submitAction(action);
  setActionButtonsEnabled(true);
}

// ── Core: submit action ────────────────────────────────────────────────────────

async function submitAction(action) {
  if (!STATE.obs) return;

  try {
    // OpenEnv V1 uses /step
    const res  = await fetch('/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(action),
    });
    const data = await res.json();
    const obs = data.observation;

    STATE.stepCount++;
    STATE.totalScore += data.reward;
    STATE.stepScores.push(data.reward);
    if (STATE.stepScores.length > MAX_CHART) STATE.stepScores.shift();

    STATE.obs = obs;

    // Update UI
    renderObs(obs);
    showLastScore(data.reward, obs.last_action_feedback);
    updateScoreDisplay();
    drawChart();

    const actionLabel = `${action.action_type}${action.esi_level ? ' ESI-' + action.esi_level : ''} → ${action.patient_id}`;
    appendLog(STATE.stepCount, action.patient_id, actionLabel, data.reward, obs.last_action_feedback);

    if (data.done) {
      appendLog(STATE.stepCount, '—', '🏁 Episode complete', null,
        `Total: ${STATE.totalScore.toFixed(2)} | Avg: ${(STATE.totalScore / STATE.stepCount).toFixed(4)}`);
      ui.stopBtn.disabled = true;
    }
  } catch (e) {
    appendLog(STATE.stepCount, '—', 'Action error: ' + e.message, null, '');
  }
}

// ── Render observation ────────────────────────────────────────────────────────

function renderObs(obs) {
  if (!obs) return;

  // Clock
  const mins = Math.floor(obs.sim_clock_minutes);
  const h    = String(Math.floor(mins / 60)).padStart(2, '0');
  const m    = String(mins % 60).padStart(2, '0');
  ui.simClock.textContent  = `${h}:${m}`;
  ui.stepCount.textContent = obs.step_count;

  // Queue meta
  const active = obs.patients.filter(p => !p.is_dispositioned).length;
  ui.queueMeta.textContent = `${active} active / ${obs.patients.length} total`;

  // Resources (hardcoded max values for now)
  updateResourceBar(ui.bedBar, ui.bedNum, obs.beds_available, 12);
  updateResourceBar(ui.labBar, ui.labNum, obs.lab_slots_available, 8);
  updateResourceBar(ui.imgBar, ui.imgNum, obs.imaging_slots_available, 4);
  updateResourceBar(ui.tokBar, ui.tokNum, obs.action_tokens_remaining, 30);

  // Patient grid
  renderPatientGrid(obs.patients);
}

function updateResourceBar(bar, numEl, current, max) {
  const pct = Math.max(0, (current / max) * 100);
  bar.style.width = pct + '%';
  numEl.textContent = `${current}/${max}`;
  bar.style.opacity = current === 0 ? '0.3' : '1';
}

function renderPatientGrid(patients) {
  if (!patients || patients.length === 0) return;

  ui.patientGrid.innerHTML = '';
  patients.forEach(p => {
    const card = buildPatientCard(p);
    ui.patientGrid.appendChild(card);
  });
}

function buildPatientCard(p) {
  const div = document.createElement('div');
  const esi  = p.assigned_esi;
  const esiClass = esi ? `esi-${esi}` : '';
  const detClass = p.has_deteriorated ? 'deteriorated' : '';

  div.className = `patient-card ${esiClass} ${detClass} ${p.is_dispositioned ? 'dispositioned' : ''} ${p.patient_id === STATE.selectedId ? 'selected' : ''}`;
  div.dataset.pid = p.patient_id;

  const badgeClass = esi ? `esi-badge esi-${esi}` : 'esi-badge unknown';
  const badgeText  = esi ? esi : '?';

  // Source Badge
  const sourceHtml = p.source === 'fhir' ? '<span class="source-badge fhir">FHIR</span>' : '';

  let vitalsHtml = '';
  if (p.heart_rate !== null && p.heart_rate !== undefined) {
    const hrClass  = p.heart_rate  > 120 ? 'abnormal' : p.heart_rate  > 100 ? 'warning' : '';
    const sbpClass = p.systolic_bp < 90  ? 'abnormal' : p.systolic_bp < 100 ? 'warning' : '';
    const spo2Class= p.spo2 < 0.90       ? 'abnormal' : p.spo2 < 0.95        ? 'warning' : '';
    const rrClass  = p.respiratory_rate > 24 ? 'abnormal' : p.respiratory_rate > 20 ? 'warning' : '';

    vitalsHtml = `
      <div class="vitals-grid">
        <div class="vital-item">
          <span class="vital-label">HR</span>
          <span class="vital-val ${hrClass}">${p.heart_rate}</span>
        </div>
        <div class="vital-item">
          <span class="vital-label">SBP</span>
          <span class="vital-val ${sbpClass}">${p.systolic_bp}</span>
        </div>
        <div class="vital-item">
          <span class="vital-label">SpO₂</span>
          <span class="vital-val ${spo2Class}">${(p.spo2 * 100).toFixed(0)}%</span>
        </div>
        <div class="vital-item">
          <span class="vital-label">RR</span>
          <span class="vital-val ${rrClass}">${p.respiratory_rate}</span>
        </div>
      </div>`;
  } else if (p.has_deteriorated) {
    vitalsHtml = `
      <div class="vitals-grid stale">
        <div class="vitals-overlay">⚠️ DATA STALE - CHECK VITALS</div>
      </div>`;
  }

  let statusHtml = '';
  if (p.is_dispositioned) {
    statusHtml = `<div class="card-status done">✓ DISPOSITIONED</div>`;
  } else if (p.has_deteriorated) {
    statusHtml = `<div class="card-status alert">🚨 STATUS CHANGE DETECTED</div>`;
  } else if (p.labs_result) {
    statusHtml = `<div class="card-status">Labs: ${p.labs_result.substring(0, 40)}…</div>`;
  }

  div.innerHTML = `
    <div class="card-top">
      <span class="patient-id">${p.patient_id} ${sourceHtml}</span>
      <span class="${badgeClass}">${badgeText}</span>
    </div>
    <div class="complaint">${p.chief_complaint}</div>
    <div class="card-meta">
      <span>⏱ ${p.wait_minutes}m</span>
      <span>${p.age_group}</span>
      <span>${p.gender}</span>
    </div>
    ${vitalsHtml}
    ${statusHtml}
  `;

  if (!p.is_dispositioned) {
    div.addEventListener('click', () => selectPatient(p.patient_id, p));
  }

  return div;
}

function selectPatient(pid, patient) {
  if (STATE.mode !== 'manual') return;
  STATE.selectedId = pid;

  // Re-render to show selection
  if (STATE.obs) renderPatientGrid(STATE.obs.patients);

  ui.selInfo.innerHTML = `
    <span>Patient </span><span class="pid">${pid}</span>
    <span style="color:var(--text-dim)"> · ${patient.chief_complaint}</span>`;

  setActionButtonsEnabled(true);
}

function setActionButtonsEnabled(enabled) {
  const hasSelection = !!STATE.selectedId;
  const hasEpisode   = !!STATE.obs && !STATE.obs.episode_done;
  const active       = enabled && hasSelection && hasEpisode;

  document.querySelectorAll('.action-btn').forEach(b => { b.disabled = !active; });
  document.querySelectorAll('.esi-btn').forEach(b   => { b.disabled = !active; });
}

// ── Score display ─────────────────────────────────────────────────────────────

function showLastScore(score, rationale) {
  ui.lastScoreBox.style.display = 'block';
  ui.lastScoreVal.textContent   = score.toFixed(4);
  ui.lastScoreBar.style.width   = (score * 100) + '%';

  const color = score >= 0.7 ? 'var(--green)' : score >= 0.4 ? 'var(--amber)' : 'var(--red)';
  ui.lastScoreVal.style.color  = color;
  ui.lastScoreBar.style.background = color;

  // Trim rationale for display
  const clean = rationale.replace(/\|.*$/, '').trim();
  ui.lastRationale.textContent = clean.length > 120 ? clean.substring(0, 120) + '…' : clean;
}

function updateScoreDisplay() {
  if (STATE.stepCount === 0) return;
  const avg = STATE.totalScore / STATE.stepCount;
  ui.bigScore.textContent = avg.toFixed(3);

  const color = avg >= 0.7 ? 'var(--green)' : avg >= 0.4 ? 'var(--amber)' : 'var(--red)';
  ui.bigScore.style.color = color;
  ui.bigScore.style.textShadow = `0 0 20px ${color}40`;
  ui.scoreSub.textContent = `avg (${STATE.stepCount} steps) · total ${STATE.totalScore.toFixed(2)}`;
}

// ── Sparkline chart ───────────────────────────────────────────────────────────

function drawChart() {
  const canvas = ui.scoreChart;
  const ctx    = canvas.getContext('2d');
  const W      = canvas.width;
  const H      = canvas.height;
  const scores = STATE.stepScores;

  ctx.clearRect(0, 0, W, H);

  if (scores.length < 2) return;

  const padX = 4;
  const padY = 6;
  const plotW = W - padX * 2;
  const plotH = H - padY * 2;

  // Background grid line at 0.5
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(padX, padY + plotH * 0.5);
  ctx.lineTo(W - padX, padY + plotH * 0.5);
  ctx.stroke();

  // Line
  ctx.beginPath();
  ctx.lineWidth   = 1.5;
  ctx.strokeStyle = '#00e676';
  ctx.shadowColor = '#00e676';
  ctx.shadowBlur  = 4;

  scores.forEach((s, i) => {
    const x = padX + (i / (scores.length - 1)) * plotW;
    const y = padY + (1 - s) * plotH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill
  ctx.shadowBlur = 0;
  const lastX = padX + plotW;
  const lastY = padY + (1 - scores[scores.length - 1]) * plotH;
  ctx.lineTo(lastX, H);
  ctx.lineTo(padX,  H);
  ctx.closePath();
  ctx.fillStyle = 'rgba(0,230,118,0.07)';
  ctx.fill();
}

// ── Episode log ───────────────────────────────────────────────────────────────

function appendLog(step, pid, action, score, rationale) {
  const empty = ui.episodeLog.querySelector('.log-empty');
  if (empty) empty.remove();

  const scoreClass = score === null ? '' : score >= 0.7 ? 'score-hi' : score >= 0.4 ? 'score-mid' : 'score-low';
  const scoreBadge = score !== null ? `<span class="log-score-badge">${score.toFixed(2)}</span>` : '';

  const entry = document.createElement('div');
  entry.className = `log-entry ${scoreClass}`;
  entry.innerHTML = `
    <span class="log-step">${step}</span>
    <div class="log-body">
      <div class="log-action">${action}</div>
      ${rationale ? `<div style="font-size:10px;color:var(--text-dim);margin-top:2px">${rationale.substring(0, 100)}</div>` : ''}
    </div>
    ${scoreBadge}
  `;

  ui.episodeLog.prepend(entry);

  const entries = ui.episodeLog.querySelectorAll('.log-entry');
  if (entries.length > MAX_LOG) entries[entries.length - 1].remove();
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

async function checkDiagnostics() {
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    ui.diagPill.className = 'diag-pill ' + (data.status === 'ok' ? 'ok' : 'degraded');
    ui.diagLabel.textContent = 'SYSTEM ONLINE';
  } catch {
    ui.diagPill.className    = 'diag-pill error';
    ui.diagLabel.textContent = 'OFFLINE';
  }
}

async function openDiagnostics() {
  ui.diagModal.style.display = 'flex';
  ui.diagContent.textContent = 'Loading…';
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    ui.diagContent.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    ui.diagContent.textContent = 'Error: ' + e.message;
  }
}
