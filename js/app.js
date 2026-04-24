/* ═══════════════════════════════════════════════════════════════
   Qynerva — Pipeline Simulation
   All MRI visuals are generated procedurally via Canvas API.
   No real patient data is used or transmitted.
═══════════════════════════════════════════════════════════════ */

// ── Helpers ──────────────────────────────────────────────────────
const sleep = ms => new Promise(r => setTimeout(r, ms));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Simple seeded pseudo-random so each slice looks consistent on reset
function seededRand(seed) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// ── Canvas: Brain MRI Slice ───────────────────────────────────────
function drawBrainSlice(canvas, opts = {}) {
  const { sliceIdx = 2, showTumor = true, seed = 42 } = opts;
  const rand = seededRand(seed + sliceIdx * 97);
  const ctx  = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const cx = w / 2;
  const cyShift = (sliceIdx - 2) * h * 0.045;
  const cy = h / 2 + cyShift;
  const decay = 1 - Math.abs(sliceIdx - 2) * 0.18;
  const skullRx = w * 0.43 * decay;
  const skullRy = h * 0.43 * decay;

  ctx.fillStyle = '#05080f';
  ctx.fillRect(0, 0, w, h);
  if (skullRy < 5) return;

  // Skull
  const sg = ctx.createRadialGradient(cx, cy, skullRy * 0.8, cx, cy, skullRx * 1.02);
  sg.addColorStop(0, '#9c9c9c'); sg.addColorStop(0.5, '#c0c0c0'); sg.addColorStop(1, '#d8d8d8');
  ctx.beginPath(); ctx.ellipse(cx, cy, skullRx, skullRy, 0, 0, Math.PI * 2);
  ctx.fillStyle = sg; ctx.fill();

  // Brain
  const bRx = skullRx * 0.86, bRy = skullRy * 0.86;
  const bg = ctx.createRadialGradient(cx - w * 0.05, cy - h * 0.06, 0, cx, cy, bRx);
  bg.addColorStop(0, '#7a7a7a'); bg.addColorStop(0.45, '#606060');
  bg.addColorStop(0.82, '#4c4c4c'); bg.addColorStop(1, '#383838');
  ctx.beginPath(); ctx.ellipse(cx, cy, bRx, bRy, 0, 0, Math.PI * 2);
  ctx.fillStyle = bg; ctx.fill();

  // Sulci (brain folds) clipped to brain ellipse
  ctx.save();
  ctx.beginPath(); ctx.ellipse(cx, cy, bRx, bRy, 0, 0, Math.PI * 2); ctx.clip();
  for (let i = 0; i < 20; i++) {
    const a  = (i / 20) * Math.PI * 2 + sliceIdx * 0.12;
    const r1 = bRx * (0.3 + rand() * 0.25);
    const r2 = bRx * (0.58 + rand() * 0.28);
    const a2 = a + 0.5 + (rand() - 0.5) * 0.3;
    const cp = a + 0.25; const cpR = bRx * (0.45 + rand() * 0.15);
    ctx.beginPath();
    ctx.moveTo(cx + r1 * Math.cos(a), cy + r1 * Math.sin(a) * (bRy / bRx));
    ctx.quadraticCurveTo(
      cx + cpR * Math.cos(cp), cy + cpR * Math.sin(cp) * (bRy / bRx),
      cx + r2 * Math.cos(a2), cy + r2 * Math.sin(a2) * (bRy / bRx)
    );
    ctx.strokeStyle = `rgba(28,28,28,${0.28 + rand() * 0.32})`;
    ctx.lineWidth = 0.7 + rand() * 0.9; ctx.stroke();
  }
  ctx.restore();

  // Ventricles (only visible in mid-slices)
  if (Math.abs(sliceIdx - 2) <= 1) {
    const vs = 1 - Math.abs(sliceIdx - 2) * 0.45;
    ctx.fillStyle = '#0c0c1e';
    ctx.beginPath(); ctx.ellipse(cx - w*0.065, cy + h*0.012, w*0.040*vs, h*0.100*vs, 0.15, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(cx + w*0.065, cy + h*0.012, w*0.040*vs, h*0.100*vs,-0.15, 0, Math.PI*2); ctx.fill();
  }

  // Tumor (bright contrast-enhancing region — left frontal)
  if (showTumor) {
    const tx = cx - w * 0.17, ty = cy - h * 0.11;
    const tg = ctx.createRadialGradient(tx, ty, 0, tx, ty, w * 0.135);
    tg.addColorStop(0,   'rgba(255,252,210,0.95)');
    tg.addColorStop(0.28,'rgba(230,210,165,0.78)');
    tg.addColorStop(0.6, 'rgba(155,125,90,0.42)');
    tg.addColorStop(1,   'rgba(80,55,30,0)');
    ctx.beginPath(); ctx.ellipse(tx, ty, w*0.135, h*0.115, 0.12, 0, Math.PI*2);
    ctx.fillStyle = tg; ctx.fill();
  }

  // Noise
  addNoise(ctx, w, h, 16, rand);
}

function addNoise(ctx, w, h, amt, rand) {
  const id = ctx.getImageData(0, 0, w, h), d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    if ((d[i] + d[i+1] + d[i+2]) / 3 < 12) continue;
    const n = (rand() - 0.5) * amt;
    d[i]   = clamp(d[i]   + n, 0, 255);
    d[i+1] = clamp(d[i+1] + n, 0, 255);
    d[i+2] = clamp(d[i+2] + n, 0, 255);
  }
  ctx.putImageData(id, 0, 0);
}

// ── Canvas: HiResCAM Heatmap Overlay ─────────────────────────────
function drawHeatmap(canvas, sliceIdx) {
  drawBrainSlice(canvas, { sliceIdx, showTumor: false, seed: 77 });
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2;
  const tx = cx - w * 0.17, ty = cy - h * 0.11;

  // Cool diffuse tint over whole brain
  const diff = ctx.createRadialGradient(cx, cy, 0, cx, cy, w * 0.38);
  diff.addColorStop(0,   'rgba(0,50,180,0.12)');
  diff.addColorStop(1,   'rgba(0,0,100,0.04)');
  ctx.beginPath(); ctx.ellipse(cx, cy, w*0.38, h*0.37, 0, 0, Math.PI*2);
  ctx.fillStyle = diff; ctx.fill();

  // Green mid-attention ring
  const mid = ctx.createRadialGradient(tx+w*0.04, ty+h*0.04, 0, tx, ty, w*0.22);
  mid.addColorStop(0,  'rgba(0,200,80,0.28)'); mid.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.beginPath(); ctx.ellipse(tx, ty, w*0.22, h*0.20, 0, 0, Math.PI*2);
  ctx.fillStyle = mid; ctx.fill();

  // Yellow warm attention
  const warm = ctx.createRadialGradient(tx, ty, 0, tx, ty, w*0.16);
  warm.addColorStop(0,  'rgba(255,225,0,0.55)');
  warm.addColorStop(0.5,'rgba(255,150,0,0.38)');
  warm.addColorStop(1,  'rgba(255,80,0,0)');
  ctx.beginPath(); ctx.ellipse(tx, ty, w*0.16, h*0.14, 0, 0, Math.PI*2);
  ctx.fillStyle = warm; ctx.fill();

  // Red hot core
  const hot = ctx.createRadialGradient(tx, ty, 0, tx, ty, w*0.09);
  hot.addColorStop(0,   'rgba(255,0,0,0.78)');
  hot.addColorStop(0.5, 'rgba(255,40,0,0.52)');
  hot.addColorStop(1,   'rgba(255,80,0,0)');
  ctx.beginPath(); ctx.ellipse(tx, ty, w*0.09, h*0.08, 0, 0, Math.PI*2);
  ctx.fillStyle = hot; ctx.fill();
}

// ── Canvas: Segmentation Mask ─────────────────────────────────────
function drawSegmentation(canvas, view = 'axial') {
  drawBrainSlice(canvas, { sliceIdx: 2, showTumor: false, seed: 55 });
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2;

  let tx, ty;
  if (view === 'axial')   { tx = cx - w*0.17; ty = cy - h*0.11; }
  else                    { tx = cx - w*0.05; ty = cy - h*0.16; }

  // ED (Edema) — yellow, outermost
  const ed = ctx.createRadialGradient(tx, ty, w*0.07, tx, ty, w*0.19);
  ed.addColorStop(0,   'rgba(251,191,36,0.48)');
  ed.addColorStop(0.5, 'rgba(245,158,11,0.26)');
  ed.addColorStop(1,   'rgba(245,158,11,0)');
  ctx.beginPath(); ctx.ellipse(tx, ty, w*0.19, h*0.165, 0, 0, Math.PI*2);
  ctx.fillStyle = ed; ctx.fill();

  // ET (Enhancing Tumor) — blue
  const et = ctx.createRadialGradient(tx, ty, 0, tx, ty, w*0.095);
  et.addColorStop(0,   'rgba(59,130,246,0.88)');
  et.addColorStop(0.6, 'rgba(59,130,246,0.52)');
  et.addColorStop(1,   'rgba(59,130,246,0)');
  ctx.beginPath(); ctx.ellipse(tx, ty, w*0.095, h*0.085, 0, 0, Math.PI*2);
  ctx.fillStyle = et; ctx.fill();

  // NCR (Necrotic Core) — red, innermost
  const ncr = ctx.createRadialGradient(tx+w*0.016, ty+h*0.012, 0, tx, ty, w*0.052);
  ncr.addColorStop(0,   'rgba(239,68,68,0.92)');
  ncr.addColorStop(0.5, 'rgba(220,38,38,0.62)');
  ncr.addColorStop(1,   'rgba(185,28,28,0)');
  ctx.beginPath(); ctx.ellipse(tx+w*0.016, ty+h*0.012, w*0.048, h*0.040, 0, 0, Math.PI*2);
  ctx.fillStyle = ncr; ctx.fill();
}

// ── UI State Helpers ──────────────────────────────────────────────
async function animateProgress(fillId, labelId, steps) {
  const fill  = document.getElementById(fillId);
  const label = document.getElementById(labelId);
  for (const { pct, text, delay } of steps) {
    fill.style.width  = pct + '%';
    label.textContent = text;
    await sleep(delay);
  }
}

function setActive(n) {
  const step   = document.getElementById(`step-${n}`);
  const status = document.getElementById(`status-${n}`);
  const body   = document.getElementById(`body-${n}`);
  step.classList.add('active');
  status.textContent = 'Processing…';
  status.className   = 'p-status st-active';
  body.style.display = 'block';
  body.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setDone(n, label) {
  const step   = document.getElementById(`step-${n}`);
  const status = document.getElementById(`status-${n}`);
  step.classList.remove('active'); step.classList.add('done');
  status.textContent = '✓ ' + label;
  status.className   = 'p-status st-done';
}

function setSkip(n, label) {
  const step   = document.getElementById(`step-${n}`);
  const status = document.getElementById(`status-${n}`);
  step.classList.add('skip');
  status.textContent = label;
  status.className   = 'p-status st-skip';
  document.getElementById(`body-${n}`).style.display = 'block';
}

function hideProgress(n) {
  const p = document.getElementById(`prog-${n}`);
  if (p) p.style.display = 'none';
  const o = document.getElementById(`out-${n}`);
  if (o) o.style.display  = 'block';
}

// Typewriter
async function typeText(el, text, speed = 9) {
  el.textContent = '';
  const cur = document.createElement('span');
  cur.className = 'cursor'; el.appendChild(cur);
  for (const ch of text) {
    cur.insertAdjacentText('beforebegin', ch);
    await sleep(speed);
  }
  cur.remove();
}

// ── Step 0: BrainMRDiff ───────────────────────────────────────────
async function runStep0() {
  setActive(0);
  await sleep(900);
  setSkip(0, 'Under Training — Bypassed');
}

// ── Step 1: Classification ────────────────────────────────────────
async function runStep1() {
  setActive(1);

  // Build slice canvases
  const row = document.getElementById('slice-row');
  row.innerHTML = '';
  const indices = [44, 63, 82, 100, 118];
  const canvases = indices.map((idx, i) => {
    const wrap = document.createElement('div'); wrap.className = 'slice-item';
    const c = document.createElement('canvas'); c.width = 108; c.height = 108;
    const lbl = document.createElement('div'); lbl.className = 'slice-lbl';
    lbl.textContent = `Slice ${idx}`;
    wrap.append(c, lbl); row.appendChild(wrap);
    return { c, idx };
  });

  await animateProgress('fill-1', 'label-1', [
    { pct: 6,  text: 'Loading NIfTI volume…',         delay: 500 },
    { pct: 12, text: 'Reorienting to axial axis…',    delay: 400 },
    { pct: 18, text: 'Extracting 156 slices…',        delay: 350 },
  ]);

  for (let i = 0; i < canvases.length; i++) {
    drawBrainSlice(canvases[i].c, { sliceIdx: i, showTumor: true, seed: 100 + i });
    document.getElementById('fill-1').style.width  = (20 + i * 14) + '%';
    document.getElementById('label-1').textContent = `Classifying slice ${canvases[i].idx} / 156…`;
    await sleep(340);
  }

  await animateProgress('fill-1', 'label-1', [
    { pct: 92,  text: 'Applying majority voting…', delay: 700 },
    { pct: 100, text: 'Classification complete.',  delay: 350 },
  ]);

  hideProgress(1);

  // Vote table
  const vt = document.getElementById('vote-table');
  vt.innerHTML = '<div class="vote-header">Slice Votes (156 total)</div>';
  const votes = [
    { cls: 'Glioma Tumor',    key: 'glioma',     n: 121, pct: 94.3 },
    { cls: 'Meningioma',      key: 'meningioma', n: 15,  pct:  9.7 },
    { cls: 'Pituitary Tumor', key: 'pituitary',  n: 12,  pct:  7.7 },
    { cls: 'Normal',          key: 'normal',     n:  8,  pct:  5.1 },
  ];
  for (const v of votes) {
    const row = document.createElement('div'); row.className = 'vote-row';
    row.innerHTML = `
      <div class="vote-cls">${v.cls}</div>
      <div class="vote-bar-wrap">
        <div class="vote-bar"><div class="vote-bar-fill ${v.key}" style="width:0%"></div></div>
        <div class="vote-count">${v.n} slices &nbsp;·&nbsp; ${v.pct}%</div>
      </div>`;
    vt.appendChild(row);
    await sleep(90);
    row.querySelector('.vote-bar-fill').style.width = ((v.n / 156) * 100) + '%';
  }

  await sleep(350);
  document.getElementById('conf-bar').style.width = '94.3%';
  setDone(1, 'Glioma Tumor — 94.3%');
}

// ── Step 2: HiResCAM ─────────────────────────────────────────────
async function runStep2() {
  setActive(2);
  await animateProgress('fill-2', 'label-2', [
    { pct: 10, text: 'Selecting top-5 confident slices…',          delay: 500 },
    { pct: 22, text: 'Registering hooks on EfficientNetB3…',       delay: 420 },
  ]);

  const grid = document.getElementById('xai-grid');
  grid.innerHTML = '';
  const slices = [63, 72, 82, 91, 100];

  for (let i = 0; i < slices.length; i++) {
    const pair = document.createElement('div'); pair.className = 'xai-pair';
    const cO = document.createElement('canvas'); cO.width = 108; cO.height = 108;
    const cH = document.createElement('canvas'); cH.width = 108; cH.height = 108;
    const lbl = document.createElement('div'); lbl.className = 'xai-lbl';
    lbl.textContent = `Slice ${slices[i]}`;
    pair.append(cO, cH, lbl); grid.appendChild(pair);

    drawBrainSlice(cO, { sliceIdx: i, showTumor: true, seed: 200 + i });

    document.getElementById('fill-2').style.width  = (28 + i * 13) + '%';
    document.getElementById('label-2').textContent = `Generating HiResCAM for slice ${slices[i]}…`;
    await sleep(480);

    drawHeatmap(cH, i);
    await sleep(80);
  }

  await animateProgress('fill-2', 'label-2', [
    { pct: 100, text: 'XAI analysis complete.', delay: 300 },
  ]);
  hideProgress(2);
  setDone(2, 'HiResCAM — 5 slices explained');
}

// ── Step 3: Segmentation ─────────────────────────────────────────
async function runStep3() {
  setActive(3);
  await animateProgress('fill-3', 'label-3', [
    { pct:  8,  text: 'Locating 4-modality patient folder…',          delay: 500 },
    { pct: 16,  text: 'Loading t1n, t1c, t2w, t2f modalities…',      delay: 600 },
    { pct: 25,  text: 'Applying MONAI preprocessing transforms…',     delay: 520 },
    { pct: 35,  text: 'Normalising per-channel (non-zero voxels)…',  delay: 440 },
    { pct: 46,  text: 'Sliding window inference — patch 1 / 12…',    delay: 600 },
    { pct: 56,  text: 'Sliding window inference — patch 4 / 12…',    delay: 540 },
    { pct: 66,  text: 'Sliding window inference — patch 8 / 12…',    delay: 540 },
    { pct: 76,  text: 'Sliding window inference — patch 12 / 12…',   delay: 540 },
    { pct: 88,  text: 'Stitching patches with Gaussian blending…',   delay: 520 },
    { pct: 95,  text: 'Remapping labels, computing volumes…',        delay: 420 },
    { pct: 100, text: 'Segmentation complete.',                       delay: 300 },
  ]);

  hideProgress(3);
  drawSegmentation(document.getElementById('seg-axial'),   'axial');
  drawSegmentation(document.getElementById('seg-coronal'), 'coronal');
  setDone(3, 'Segmentation — 32,620 mm³ total');
}

// ── Step 4: Clinical Report ───────────────────────────────────────
async function runStep4() {
  setActive(4);
  await animateProgress('fill-4', 'label-4', [
    { pct: 15,  text: 'Bundling pipeline findings…',                       delay: 500 },
    { pct: 35,  text: 'Building structured prompt…',                       delay: 420 },
    { pct: 55,  text: 'Sending to Mistral AI (mistral-large-latest)…',    delay: 850 },
    { pct: 82,  text: 'Receiving clinical narrative…',                     delay: 1100 },
    { pct: 100, text: 'Report generated.',                                 delay: 320 },
  ]);

  hideProgress(4);

  const today = new Date().toISOString().slice(0, 10);
  const report = `══════════════════════════════════════════════════════════
  QYNERVA — AI-ASSISTED BRAIN TUMOR ANALYSIS REPORT
══════════════════════════════════════════════════════════

1. PATIENT INFORMATION
   Scan file  : BraTS-GLI-00006-101-t1c.nii.gz
   Modality   : T1 Contrast-Enhanced
   Analysis   : ${today}

2. DIAGNOSIS SUMMARY
   Predicted class : Glioma Tumor
   Confidence      : 94.3%
   Votes           : 121 / 156 slices

3. MODEL CONFIDENCE ANALYSIS
   The model assigned 94.3% probability to glioma_tumor, with
   the remaining probability distributed across meningioma (9.7%),
   pituitary (7.7%), and normal (5.1%). The high margin between
   the top class and the runner-up indicates a reliable prediction.

4. EXPLAINABILITY FINDINGS  (HiResCAM)
   Analysis of the top 5 most confident slices shows consistent
   activation in the central-left hemisphere region, corresponding
   to typical glioma presentation in the frontal lobe. Attention
   patterns were stable across all slices, suggesting the model
   is responding to genuine tumour signal rather than imaging
   artefacts or patient motion.

5. SEGMENTATION SUMMARY  (3D U-Net)
   Necrotic Core  (NCR) :  4,210 mm³   [red   ]
   Peritumoral Edema (ED): 18,540 mm³   [yellow]
   Enhancing Tumor  (ET) :  9,870 mm³   [blue  ]
   ─────────────────────────────────────────────
   Total tumour volume   : 32,620 mm³

   The enhancing tumour component represents 30.3% of total
   tumour volume, consistent with an active, high-grade glioma
   (WHO Grade IV). The large peritumoral oedema volume indicates
   significant mass effect on surrounding brain tissue.

6. CLINICAL INTERPRETATION & DISCLAIMER
   These findings are generated by an automated AI pipeline and
   are intended as a decision-support aid only. All results must
   be reviewed and validated by a qualified radiologist or
   neuro-oncologist before any clinical action is taken.
   This report does not constitute a medical diagnosis.

══════════════════════════════════════════════════════════`;

  await typeText(document.getElementById('report-text'), report, 9);
  setDone(4, 'Clinical Report Generated');
}

// ── Main Orchestrator ─────────────────────────────────────────────
let running = false;

async function startPipeline() {
  if (running) return;
  running = true;

  document.getElementById('load-btn').disabled = true;
  document.getElementById('upload-zone').style.display = 'none';

  const scanLoaded = document.getElementById('scan-loaded');
  scanLoaded.style.display = 'flex';
  const preview = document.getElementById('scan-preview');
  drawBrainSlice(preview, { sliceIdx: 2, showTumor: true, seed: 42 });

  const stepsEl = document.getElementById('pipeline-steps');
  stepsEl.style.display = 'block';

  await sleep(500);
  await runStep0();
  await sleep(350);
  await runStep1();
  await sleep(350);
  await runStep2();
  await sleep(350);
  await runStep3();
  await sleep(350);
  await runStep4();
  await sleep(400);

  const done = document.getElementById('pipeline-complete');
  done.style.display = 'block';
  done.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetPipeline() {
  running = false;

  document.getElementById('upload-zone').style.display = 'block';
  document.getElementById('load-btn').disabled = false;
  document.getElementById('scan-loaded').style.display = 'none';
  document.getElementById('pipeline-steps').style.display = 'none';
  document.getElementById('pipeline-complete').style.display = 'none';

  for (let n = 0; n <= 4; n++) {
    const step = document.getElementById(`step-${n}`);
    if (step) step.classList.remove('active','done','skip');

    const status = document.getElementById(`status-${n}`);
    if (status) { status.textContent = 'Waiting'; status.className = 'p-status'; }

    const body = document.getElementById(`body-${n}`);
    if (body) body.style.display = 'none';

    const fill = document.getElementById(`fill-${n}`);
    if (fill) fill.style.width = '0%';

    const label = document.getElementById(`label-${n}`);
    if (label) label.textContent = 'Initialising…';

    const prog = document.getElementById(`prog-${n}`);
    if (prog) prog.style.display = 'block';

    const out = document.getElementById(`out-${n}`);
    if (out) out.style.display = 'none';
  }

  // Clear dynamic content
  const ids = ['slice-row','vote-table','xai-grid','report-text'];
  ids.forEach(id => { const el = document.getElementById(id); if (el) el.innerHTML = ''; });
  const cb = document.getElementById('conf-bar');
  if (cb) cb.style.width = '0%';
}

// ── Scroll header shadow ─────────────────────────────────────────
window.addEventListener('scroll', () => {
  document.getElementById('header').style.boxShadow =
    window.scrollY > 8 ? '0 4px 24px rgba(0,0,0,0.45)' : 'none';
});
