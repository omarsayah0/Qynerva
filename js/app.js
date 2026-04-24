/*
  Qynerva - Pipeline Simulation
  All MRI visuals are generated procedurally via Canvas API.
  No real patient data is used or transmitted.
*/

// Helpers
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function seededRand(seed) {
  let state = seed;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function getSliceGeometry(width, height, sliceIdx = 2) {
  const cx = width / 2;
  const cy = height / 2 + (sliceIdx - 2) * height * 0.038;
  const depth = Math.max(0.35, 1 - Math.abs(sliceIdx - 2) * 0.17);
  const brainRx = width * 0.29 * depth;
  const brainRy = height * 0.34 * depth;

  return {
    cx,
    cy,
    depth,
    brainRx,
    brainRy,
    skullRx: brainRx * 1.12,
    skullRy: brainRy * 1.12,
    tumorX: cx - brainRx * 0.40,
    tumorY: cy - brainRy * 0.28
  };
}

function traceAxialBrainPath(ctx, cx, cy, rx, ry) {
  ctx.beginPath();
  ctx.moveTo(cx, cy - ry);
  ctx.bezierCurveTo(
    cx + rx * 0.42, cy - ry * 1.05,
    cx + rx * 0.94, cy - ry * 0.48,
    cx + rx * 0.95, cy + ry * 0.06
  );
  ctx.bezierCurveTo(
    cx + rx * 0.98, cy + ry * 0.48,
    cx + rx * 0.57, cy + ry * 0.94,
    cx + rx * 0.14, cy + ry * 0.99
  );
  ctx.quadraticCurveTo(cx, cy + ry * 1.03, cx - rx * 0.14, cy + ry * 0.99);
  ctx.bezierCurveTo(
    cx - rx * 0.57, cy + ry * 0.94,
    cx - rx * 0.98, cy + ry * 0.48,
    cx - rx * 0.95, cy + ry * 0.06
  );
  ctx.bezierCurveTo(
    cx - rx * 0.94, cy - ry * 0.48,
    cx - rx * 0.42, cy - ry * 1.05,
    cx, cy - ry
  );
  ctx.closePath();
}

// Canvas: Brain MRI slice
function drawBrainSlice(canvas, opts = {}) {
  const { sliceIdx = 2, showTumor = true, seed = 42 } = opts;
  const rand = seededRand(seed + sliceIdx * 97);
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const { cx, cy, depth, brainRx, brainRy, skullRx, skullRy, tumorX, tumorY } = getSliceGeometry(width, height, sliceIdx);

  ctx.clearRect(0, 0, width, height);

  const background = ctx.createRadialGradient(cx, cy, width * 0.04, cx, cy, width * 0.78);
  background.addColorStop(0, '#0b1220');
  background.addColorStop(0.58, '#060b15');
  background.addColorStop(1, '#010308');
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  if (brainRy < 5) return;

  ctx.strokeStyle = 'rgba(130, 180, 255, 0.06)';
  ctx.lineWidth = 1;
  ctx.strokeRect(1.5, 1.5, width - 3, height - 3);

  traceAxialBrainPath(ctx, cx, cy, skullRx * 1.03, skullRy * 1.04);
  const scalp = ctx.createRadialGradient(cx, cy - skullRy * 0.2, 0, cx, cy, skullRx * 1.2);
  scalp.addColorStop(0, 'rgba(150, 156, 170, 0.14)');
  scalp.addColorStop(0.72, 'rgba(80, 84, 94, 0.10)');
  scalp.addColorStop(1, 'rgba(30, 32, 40, 0.02)');
  ctx.fillStyle = scalp;
  ctx.fill();

  traceAxialBrainPath(ctx, cx, cy, skullRx, skullRy);
  ctx.lineWidth = Math.max(1.4, width * 0.017);
  ctx.strokeStyle = 'rgba(195, 202, 214, 0.16)';
  ctx.stroke();

  traceAxialBrainPath(ctx, cx, cy, brainRx, brainRy);
  const tissue = ctx.createRadialGradient(cx, cy - brainRy * 0.48, 0, cx, cy + brainRy * 0.05, brainRx * 1.18);
  tissue.addColorStop(0, '#8a9098');
  tissue.addColorStop(0.35, '#636b75');
  tissue.addColorStop(0.72, '#353c46');
  tissue.addColorStop(1, '#191f28');
  ctx.fillStyle = tissue;
  ctx.fill();

  ctx.save();
  traceAxialBrainPath(ctx, cx, cy, brainRx, brainRy);
  ctx.clip();

  const leftMatter = ctx.createRadialGradient(cx - brainRx * 0.24, cy - brainRy * 0.08, 0, cx - brainRx * 0.14, cy, brainRx * 0.62);
  leftMatter.addColorStop(0, 'rgba(216, 219, 224, 0.28)');
  leftMatter.addColorStop(1, 'rgba(255, 255, 255, 0)');
  ctx.fillStyle = leftMatter;
  ctx.beginPath();
  ctx.ellipse(cx - brainRx * 0.16, cy + brainRy * 0.06, brainRx * 0.48, brainRy * 0.62, -0.18, 0, Math.PI * 2);
  ctx.fill();

  const rightMatter = ctx.createRadialGradient(cx + brainRx * 0.2, cy - brainRy * 0.04, 0, cx + brainRx * 0.12, cy, brainRx * 0.58);
  rightMatter.addColorStop(0, 'rgba(205, 209, 216, 0.24)');
  rightMatter.addColorStop(1, 'rgba(255, 255, 255, 0)');
  ctx.fillStyle = rightMatter;
  ctx.beginPath();
  ctx.ellipse(cx + brainRx * 0.15, cy + brainRy * 0.05, brainRx * 0.46, brainRy * 0.6, 0.18, 0, Math.PI * 2);
  ctx.fill();

  ctx.beginPath();
  ctx.ellipse(cx, cy + brainRy * 0.1, brainRx * 0.34, brainRy * 0.46, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(232, 236, 242, 0.10)';
  ctx.fill();

  for (const side of [-1, 1]) {
    for (let index = 0; index < 8; index++) {
      const y = cy - brainRy * 0.68 + index * brainRy * 0.18 + (rand() - 0.5) * brainRy * 0.04;
      const edgeX = cx + side * brainRx * (0.78 - rand() * 0.08);
      const midX = cx + side * brainRx * (0.40 + rand() * 0.08);
      const endX = cx + side * brainRx * (0.08 + rand() * 0.08);
      ctx.beginPath();
      ctx.moveTo(edgeX, y);
      ctx.bezierCurveTo(
        cx + side * brainRx * (0.60 + rand() * 0.04), y - brainRy * (0.07 + rand() * 0.04),
        midX, y + brainRy * (0.04 + rand() * 0.05),
        endX, y + brainRy * (0.08 + rand() * 0.03)
      );
      ctx.strokeStyle = `rgba(12, 16, 24, ${0.20 + rand() * 0.16})`;
      ctx.lineWidth = 0.8 + rand() * 0.65;
      ctx.stroke();
    }
  }

  ctx.beginPath();
  ctx.moveTo(cx, cy - brainRy * 0.82);
  ctx.quadraticCurveTo(cx - brainRx * 0.02, cy - brainRy * 0.08, cx, cy + brainRy * 0.88);
  ctx.strokeStyle = 'rgba(236, 240, 246, 0.12)';
  ctx.lineWidth = Math.max(1, width * 0.008);
  ctx.stroke();

  const ventricleScale = Math.max(0, depth - 0.34);
  if (ventricleScale > 0) {
    ctx.fillStyle = `rgba(6, 10, 18, ${0.60 * ventricleScale})`;
    ctx.beginPath();
    ctx.ellipse(cx - brainRx * 0.17, cy + brainRy * 0.03, brainRx * 0.10 * ventricleScale, brainRy * 0.22 * ventricleScale, 0.42, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(cx + brainRx * 0.17, cy + brainRy * 0.03, brainRx * 0.10 * ventricleScale, brainRy * 0.22 * ventricleScale, -0.42, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(cx, cy + brainRy * 0.14, brainRx * 0.08 * ventricleScale, brainRy * 0.09 * ventricleScale, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  if (showTumor) {
    const tumorScale = Math.max(0.46, 1 - Math.abs(sliceIdx - 2) * 0.2);

    const edema = ctx.createRadialGradient(tumorX, tumorY, 0, tumorX, tumorY, brainRx * 0.36);
    edema.addColorStop(0, `rgba(155, 130, 108, ${0.28 * tumorScale})`);
    edema.addColorStop(0.6, `rgba(95, 72, 54, ${0.16 * tumorScale})`);
    edema.addColorStop(1, 'rgba(45, 28, 18, 0)');
    ctx.fillStyle = edema;
    ctx.beginPath();
    ctx.ellipse(tumorX - brainRx * 0.02, tumorY + brainRy * 0.02, brainRx * 0.28 * tumorScale, brainRy * 0.22 * tumorScale, 0.15, 0, Math.PI * 2);
    ctx.fill();

    const rim = ctx.createRadialGradient(tumorX, tumorY, brainRx * 0.02, tumorX, tumorY, brainRx * 0.18);
    rim.addColorStop(0, `rgba(255, 246, 220, ${0.96 * tumorScale})`);
    rim.addColorStop(0.35, `rgba(245, 218, 176, ${0.84 * tumorScale})`);
    rim.addColorStop(0.64, `rgba(170, 130, 92, ${0.38 * tumorScale})`);
    rim.addColorStop(1, 'rgba(70, 42, 22, 0)');
    ctx.fillStyle = rim;
    ctx.beginPath();
    ctx.ellipse(tumorX, tumorY, brainRx * 0.18 * tumorScale, brainRy * 0.15 * tumorScale, 0.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.ellipse(tumorX + brainRx * 0.03, tumorY + brainRy * 0.01, brainRx * 0.07 * tumorScale, brainRy * 0.055 * tumorScale, 0.1, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(32, 24, 20, ${0.56 * tumorScale})`;
    ctx.fill();

    ctx.beginPath();
    ctx.ellipse(tumorX - brainRx * 0.04, tumorY - brainRy * 0.02, brainRx * 0.05 * tumorScale, brainRy * 0.04 * tumorScale, -0.3, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255, 250, 235, ${0.75 * tumorScale})`;
    ctx.fill();
  }

  ctx.restore();

  const vignette = ctx.createRadialGradient(cx, cy, brainRx * 0.55, cx, cy, width * 0.72);
  vignette.addColorStop(0, 'rgba(0, 0, 0, 0)');
  vignette.addColorStop(1, 'rgba(0, 0, 0, 0.42)');
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, width, height);

  addNoise(ctx, width, height, 10, rand);
}

function addNoise(ctx, width, height, amount, rand) {
  const imageData = ctx.getImageData(0, 0, width, height);
  const pixels = imageData.data;

  for (let offset = 0; offset < pixels.length; offset += 4) {
    if ((pixels[offset] + pixels[offset + 1] + pixels[offset + 2]) / 3 < 12) continue;
    const noise = (rand() - 0.5) * amount;
    pixels[offset] = clamp(pixels[offset] + noise, 0, 255);
    pixels[offset + 1] = clamp(pixels[offset + 1] + noise, 0, 255);
    pixels[offset + 2] = clamp(pixels[offset + 2] + noise, 0, 255);
  }

  ctx.putImageData(imageData, 0, 0);
}

// Canvas: HiResCAM heatmap overlay
function drawHeatmap(canvas, sliceIdx) {
  drawBrainSlice(canvas, { sliceIdx, showTumor: false, seed: 77 });
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const { cx, cy, brainRx, brainRy, tumorX, tumorY } = getSliceGeometry(width, height, sliceIdx);

  const diffuse = ctx.createRadialGradient(cx, cy, 0, cx, cy, brainRx * 1.15);
  diffuse.addColorStop(0, 'rgba(0,50,180,0.12)');
  diffuse.addColorStop(1, 'rgba(0,0,100,0.04)');
  traceAxialBrainPath(ctx, cx, cy, brainRx, brainRy);
  ctx.fillStyle = diffuse;
  ctx.fill();

  const mid = ctx.createRadialGradient(tumorX + brainRx * 0.12, tumorY + brainRy * 0.10, 0, tumorX, tumorY, brainRx * 0.42);
  mid.addColorStop(0, 'rgba(0,200,80,0.28)');
  mid.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.beginPath();
  ctx.ellipse(tumorX, tumorY, brainRx * 0.42, brainRy * 0.32, 0, 0, Math.PI * 2);
  ctx.fillStyle = mid;
  ctx.fill();

  const warm = ctx.createRadialGradient(tumorX, tumorY, 0, tumorX, tumorY, brainRx * 0.28);
  warm.addColorStop(0, 'rgba(255,225,0,0.55)');
  warm.addColorStop(0.5, 'rgba(255,150,0,0.38)');
  warm.addColorStop(1, 'rgba(255,80,0,0)');
  ctx.beginPath();
  ctx.ellipse(tumorX, tumorY, brainRx * 0.28, brainRy * 0.22, 0, 0, Math.PI * 2);
  ctx.fillStyle = warm;
  ctx.fill();

  const hot = ctx.createRadialGradient(tumorX, tumorY, 0, tumorX, tumorY, brainRx * 0.16);
  hot.addColorStop(0, 'rgba(255,0,0,0.78)');
  hot.addColorStop(0.5, 'rgba(255,40,0,0.52)');
  hot.addColorStop(1, 'rgba(255,80,0,0)');
  ctx.beginPath();
  ctx.ellipse(tumorX, tumorY, brainRx * 0.16, brainRy * 0.12, 0, 0, Math.PI * 2);
  ctx.fillStyle = hot;
  ctx.fill();
}

// Canvas: Segmentation mask
function getVirtualSliceIndex(progress) {
  return 2 + (progress - 0.5) * 3.3;
}

function getSegmentationProfile(view, progress, width, height) {
  const cfg = {
    axial: {
      center: 0.58,
      spread: 0.26,
      x: 0.33,
      y: 0.40,
      driftX: 0.03,
      driftY: 0.02,
      rotate: 0.12,
      sizeX: 0.19,
      sizeY: 0.165
    },
    coronal: {
      center: 0.52,
      spread: 0.30,
      x: 0.46,
      y: 0.34,
      driftX: 0.02,
      driftY: 0.05,
      rotate: -0.08,
      sizeX: 0.17,
      sizeY: 0.22
    },
    sagittal: {
      center: 0.45,
      spread: 0.24,
      x: 0.54,
      y: 0.40,
      driftX: 0.05,
      driftY: 0.03,
      rotate: 0.18,
      sizeX: 0.14,
      sizeY: 0.20
    }
  }[view] || {
    center: 0.58,
    spread: 0.26,
    x: 0.33,
    y: 0.40,
    driftX: 0.03,
    driftY: 0.02,
    rotate: 0.12,
    sizeX: 0.19,
    sizeY: 0.165
  };

  const falloff = Math.max(0, 1 - Math.abs(progress - cfg.center) / cfg.spread);
  const intensity = Math.pow(falloff, 1.35);
  const shift = progress - cfg.center;

  return {
    intensity,
    tx: width * (cfg.x + shift * cfg.driftX),
    ty: height * (cfg.y + shift * cfg.driftY),
    sizeX: width * cfg.sizeX * (0.62 + intensity * 0.58),
    sizeY: height * cfg.sizeY * (0.62 + intensity * 0.58),
    rotation: cfg.rotate + shift * 0.55
  };
}

function drawSegmentation(canvas, view = 'axial', sliceProgress = 0.58) {
  drawBrainSlice(canvas, {
    sliceIdx: getVirtualSliceIndex(sliceProgress),
    showTumor: false,
    seed: 55 + Math.round(sliceProgress * 20) + view.length * 7
  });
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const profile = getSegmentationProfile(view, sliceProgress, width, height);
  if (profile.intensity <= 0.03) return;

  const tx = profile.tx;
  const ty = profile.ty;
  const edema = ctx.createRadialGradient(tx, ty, width * 0.04, tx, ty, profile.sizeX);
  edema.addColorStop(0, 'rgba(251,191,36,0.48)');
  edema.addColorStop(0.5, `rgba(245,158,11,${0.16 + profile.intensity * 0.2})`);
  edema.addColorStop(1, 'rgba(245,158,11,0)');
  ctx.beginPath();
  ctx.ellipse(tx, ty, profile.sizeX, profile.sizeY, profile.rotation, 0, Math.PI * 2);
  ctx.fillStyle = edema;
  ctx.fill();

  const et = ctx.createRadialGradient(tx, ty, 0, tx, ty, profile.sizeX * 0.48);
  et.addColorStop(0, 'rgba(59,130,246,0.88)');
  et.addColorStop(0.6, `rgba(59,130,246,${0.32 + profile.intensity * 0.28})`);
  et.addColorStop(1, 'rgba(59,130,246,0)');
  ctx.beginPath();
  ctx.ellipse(tx, ty, profile.sizeX * 0.5, profile.sizeY * 0.5, profile.rotation * 0.9, 0, Math.PI * 2);
  ctx.fillStyle = et;
  ctx.fill();

  const ncr = ctx.createRadialGradient(tx + width * 0.016, ty + height * 0.012, 0, tx, ty, profile.sizeX * 0.28);
  ncr.addColorStop(0, 'rgba(239,68,68,0.92)');
  ncr.addColorStop(0.5, `rgba(220,38,38,${0.4 + profile.intensity * 0.22})`);
  ncr.addColorStop(1, 'rgba(185,28,28,0)');
  ctx.beginPath();
  ctx.ellipse(
    tx + profile.sizeX * 0.09,
    ty + profile.sizeY * 0.05,
    profile.sizeX * 0.26,
    profile.sizeY * 0.24,
    profile.rotation * 0.8,
    0,
    Math.PI * 2
  );
  ctx.fillStyle = ncr;
  ctx.fill();
}

const segmentationViewerState = {
  plane: 'axial',
  slice: 96,
  maxSlice: 154,
  playing: false,
  timer: null,
  setup: false
};

function renderSegmentationExplorer() {
  const canvas = document.getElementById('seg-slice-viewer');
  const range = document.getElementById('seg-slice-range');
  const label = document.getElementById('seg-slice-label');
  const playBtn = document.getElementById('seg-play-btn');
  const planeReadout = document.getElementById('seg-plane-readout');

  if (!canvas || !range || !label || !playBtn || !planeReadout) return;

  range.max = segmentationViewerState.maxSlice;
  range.value = segmentationViewerState.slice;

  const sliceDisplay = segmentationViewerState.slice + 1;
  const planeLabel = segmentationViewerState.plane.charAt(0).toUpperCase() + segmentationViewerState.plane.slice(1);
  label.textContent = `Slice ${sliceDisplay} / ${segmentationViewerState.maxSlice + 1}`;
  planeReadout.textContent = `${planeLabel} stack`;
  playBtn.textContent = segmentationViewerState.playing ? 'Pause Auto Slice' : 'Play Auto Slice';

  document.querySelectorAll('.seg-plane-btn').forEach(button => {
    button.classList.toggle('active', button.dataset.plane === segmentationViewerState.plane);
  });

  drawSegmentation(
    canvas,
    segmentationViewerState.plane,
    segmentationViewerState.slice / segmentationViewerState.maxSlice
  );
}

function stopSegmentationPlayback() {
  if (segmentationViewerState.timer) {
    clearInterval(segmentationViewerState.timer);
    segmentationViewerState.timer = null;
  }
  segmentationViewerState.playing = false;
  renderSegmentationExplorer();
}

function startSegmentationPlayback() {
  if (segmentationViewerState.timer) return;
  segmentationViewerState.playing = true;
  segmentationViewerState.timer = setInterval(() => {
    segmentationViewerState.slice = (segmentationViewerState.slice + 1) % (segmentationViewerState.maxSlice + 1);
    renderSegmentationExplorer();
  }, 120);
  renderSegmentationExplorer();
}

function setupSegmentationExplorer() {
  if (segmentationViewerState.setup) return;

  const range = document.getElementById('seg-slice-range');
  const playBtn = document.getElementById('seg-play-btn');
  const planeButtons = document.querySelectorAll('.seg-plane-btn');
  if (!range || !playBtn || !planeButtons.length) return;

  segmentationViewerState.setup = true;

  range.addEventListener('input', event => {
    segmentationViewerState.slice = Number(event.target.value);
    renderSegmentationExplorer();
  });

  playBtn.addEventListener('click', () => {
    if (segmentationViewerState.playing) stopSegmentationPlayback();
    else startSegmentationPlayback();
  });

  planeButtons.forEach(button => {
    button.addEventListener('click', () => {
      segmentationViewerState.plane = button.dataset.plane;
      renderSegmentationExplorer();
    });
  });
}

// UI state helpers
async function animateProgress(fillId, labelId, steps) {
  const fill = document.getElementById(fillId);
  const label = document.getElementById(labelId);
  for (const { pct, text, delay } of steps) {
    fill.style.width = `${pct}%`;
    label.textContent = text;
    await sleep(delay);
  }
}

function setActive(stepNum) {
  const step = document.getElementById(`step-${stepNum}`);
  const status = document.getElementById(`status-${stepNum}`);
  const body = document.getElementById(`body-${stepNum}`);
  step.classList.add('active');
  status.textContent = 'Processing...';
  status.className = 'p-status st-active';
  body.style.display = 'block';
  body.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setDone(stepNum, label) {
  const step = document.getElementById(`step-${stepNum}`);
  const status = document.getElementById(`status-${stepNum}`);
  step.classList.remove('active');
  step.classList.add('done');
  status.textContent = `OK ${label}`;
  status.className = 'p-status st-done';
}

function setSkip(stepNum, label) {
  const step = document.getElementById(`step-${stepNum}`);
  const status = document.getElementById(`status-${stepNum}`);
  step.classList.add('skip');
  status.textContent = label;
  status.className = 'p-status st-skip';
  document.getElementById(`body-${stepNum}`).style.display = 'block';
}

function hideProgress(stepNum) {
  const progress = document.getElementById(`prog-${stepNum}`);
  if (progress) progress.style.display = 'none';
  const output = document.getElementById(`out-${stepNum}`);
  if (output) output.style.display = 'block';
}

async function typeText(element, text, speed = 9) {
  element.textContent = '';
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  element.appendChild(cursor);
  for (const char of text) {
    cursor.insertAdjacentText('beforebegin', char);
    await sleep(speed);
  }
  cursor.remove();
}

// Step 0: BrainMRDiff
async function runStep0() {
  setActive(0);
  await sleep(900);
  setSkip(0, 'Under Training - Bypassed');
}

// Step 1: Classification
async function runStep1() {
  setActive(1);

  const row = document.getElementById('slice-row');
  row.innerHTML = '';
  const indices = [44, 63, 82, 100, 118];
  const canvases = indices.map((idx, displayIdx) => {
    const wrap = document.createElement('div');
    wrap.className = 'slice-item';
    const canvas = document.createElement('canvas');
    canvas.width = 108;
    canvas.height = 108;
    const label = document.createElement('div');
    label.className = 'slice-lbl';
    label.textContent = `Slice ${idx}`;
    wrap.append(canvas, label);
    row.appendChild(wrap);
    return { canvas, idx, displayIdx };
  });

  await animateProgress('fill-1', 'label-1', [
    { pct: 6, text: 'Loading NIfTI volume...', delay: 500 },
    { pct: 12, text: 'Reorienting to axial axis...', delay: 400 },
    { pct: 18, text: 'Extracting 156 slices...', delay: 350 }
  ]);

  for (let index = 0; index < canvases.length; index++) {
    const item = canvases[index];
    drawBrainSlice(item.canvas, { sliceIdx: item.displayIdx, showTumor: true, seed: 100 + index });
    document.getElementById('fill-1').style.width = `${20 + index * 14}%`;
    document.getElementById('label-1').textContent = `Classifying slice ${item.idx} / 156...`;
    await sleep(340);
  }

  await animateProgress('fill-1', 'label-1', [
    { pct: 92, text: 'Applying majority voting...', delay: 700 },
    { pct: 100, text: 'Classification complete.', delay: 350 }
  ]);

  hideProgress(1);

  const voteTable = document.getElementById('vote-table');
  voteTable.innerHTML = '<div class="vote-header">Slice Votes (156 total)</div>';
  const votes = [
    { cls: 'Glioma Tumor', key: 'glioma', n: 121, pct: 94.3 },
    { cls: 'Meningioma', key: 'meningioma', n: 15, pct: 9.7 },
    { cls: 'Pituitary Tumor', key: 'pituitary', n: 12, pct: 7.7 },
    { cls: 'Normal', key: 'normal', n: 8, pct: 5.1 }
  ];

  for (const vote of votes) {
    const rowElement = document.createElement('div');
    rowElement.className = 'vote-row';
    rowElement.innerHTML = `
      <div class="vote-cls">${vote.cls}</div>
      <div class="vote-bar-wrap">
        <div class="vote-bar"><div class="vote-bar-fill ${vote.key}" style="width:0%"></div></div>
        <div class="vote-count">${vote.n} slices · ${vote.pct}%</div>
      </div>`;
    voteTable.appendChild(rowElement);
    await sleep(90);
    rowElement.querySelector('.vote-bar-fill').style.width = `${(vote.n / 156) * 100}%`;
  }

  await sleep(350);
  document.getElementById('conf-bar').style.width = '94.3%';
  setDone(1, 'Glioma Tumor - 94.3%');
}

// Step 2: HiResCAM
async function runStep2() {
  setActive(2);
  await animateProgress('fill-2', 'label-2', [
    { pct: 10, text: 'Selecting top-5 confident slices...', delay: 500 },
    { pct: 22, text: 'Registering hooks on EfficientNetB3...', delay: 420 }
  ]);

  const grid = document.getElementById('xai-grid');
  grid.innerHTML = '';
  const slices = [63, 72, 82, 91, 100];

  for (let index = 0; index < slices.length; index++) {
    const pair = document.createElement('div');
    pair.className = 'xai-pair';
    const original = document.createElement('canvas');
    original.width = 108;
    original.height = 108;
    const heatmap = document.createElement('canvas');
    heatmap.width = 108;
    heatmap.height = 108;
    const label = document.createElement('div');
    label.className = 'xai-lbl';
    label.textContent = `Slice ${slices[index]}`;
    pair.append(original, heatmap, label);
    grid.appendChild(pair);

    drawBrainSlice(original, { sliceIdx: index, showTumor: true, seed: 200 + index });

    document.getElementById('fill-2').style.width = `${28 + index * 13}%`;
    document.getElementById('label-2').textContent = `Generating HiResCAM for slice ${slices[index]}...`;
    await sleep(480);

    drawHeatmap(heatmap, index);
    await sleep(80);
  }

  await animateProgress('fill-2', 'label-2', [
    { pct: 100, text: 'XAI analysis complete.', delay: 300 }
  ]);
  hideProgress(2);
  setDone(2, 'HiResCAM - 5 slices explained');
}

// Step 3: Segmentation
async function runStep3() {
  setActive(3);
  await animateProgress('fill-3', 'label-3', [
    { pct: 8, text: 'Locating 4-modality patient folder...', delay: 500 },
    { pct: 16, text: 'Loading t1n, t1c, t2w, t2f modalities...', delay: 600 },
    { pct: 25, text: 'Applying MONAI preprocessing transforms...', delay: 520 },
    { pct: 35, text: 'Normalising per-channel (non-zero voxels)...', delay: 440 },
    { pct: 46, text: 'Sliding window inference - patch 1 / 12...', delay: 600 },
    { pct: 56, text: 'Sliding window inference - patch 4 / 12...', delay: 540 },
    { pct: 66, text: 'Sliding window inference - patch 8 / 12...', delay: 540 },
    { pct: 76, text: 'Sliding window inference - patch 12 / 12...', delay: 540 },
    { pct: 88, text: 'Stitching patches with Gaussian blending...', delay: 520 },
    { pct: 95, text: 'Remapping labels, computing volumes...', delay: 420 },
    { pct: 100, text: 'Segmentation complete.', delay: 300 }
  ]);

  hideProgress(3);
  setupSegmentationExplorer();
  drawSegmentation(document.getElementById('seg-axial'), 'axial', 0.58);
  drawSegmentation(document.getElementById('seg-coronal'), 'coronal', 0.52);
  segmentationViewerState.slice = 96;
  segmentationViewerState.plane = 'axial';
  renderSegmentationExplorer();
  startSegmentationPlayback();
  setDone(3, 'Segmentation - 32,620 mm3 total');
}

// Step 4: Clinical report
async function runStep4() {
  setActive(4);
  await animateProgress('fill-4', 'label-4', [
    { pct: 15, text: 'Bundling pipeline findings...', delay: 500 },
    { pct: 35, text: 'Building structured prompt...', delay: 420 },
    { pct: 55, text: 'Sending to Mistral AI (mistral-large-latest)...', delay: 850 },
    { pct: 82, text: 'Receiving clinical narrative...', delay: 1100 },
    { pct: 100, text: 'Report generated.', delay: 320 }
  ]);

  hideProgress(4);

  const today = new Date().toISOString().slice(0, 10);
  const report = `==========================================================
  QYNERVA - AI-ASSISTED BRAIN TUMOR ANALYSIS REPORT
==========================================================

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

4. EXPLAINABILITY FINDINGS (HiResCAM)
   Analysis of the top 5 most confident slices shows consistent
   activation in the central-left hemisphere region, corresponding
   to typical glioma presentation in the frontal lobe. Attention
   patterns were stable across all slices, suggesting the model
   is responding to genuine tumour signal rather than imaging
   artefacts or patient motion.

5. SEGMENTATION SUMMARY (3D U-Net)
   Necrotic Core (NCR)    :  4,210 mm3   [red]
   Peritumoral Edema (ED) : 18,540 mm3   [yellow]
   Enhancing Tumor (ET)   :  9,870 mm3   [blue]
   ---------------------------------------------
   Total tumour volume    : 32,620 mm3

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

==========================================================`;

  await typeText(document.getElementById('report-text'), report, 9);
  setDone(4, 'Clinical Report Generated');
}

// Main orchestrator
let running = false;

function renderStaticPreviews() {
  const uploadPreview = document.getElementById('upload-preview');
  if (uploadPreview) {
    drawBrainSlice(uploadPreview, { sliceIdx: 2, showTumor: true, seed: 42 });
  }

  const scanPreview = document.getElementById('scan-preview');
  if (scanPreview) {
    drawBrainSlice(scanPreview, { sliceIdx: 2, showTumor: true, seed: 42 });
  }

  setupSegmentationExplorer();
  renderSegmentationExplorer();
}

async function startPipeline() {
  if (running) return;
  running = true;

  document.getElementById('load-btn').disabled = true;
  document.getElementById('upload-zone').style.display = 'none';

  const scanLoaded = document.getElementById('scan-loaded');
  scanLoaded.style.display = 'flex';
  const preview = document.getElementById('scan-preview');
  drawBrainSlice(preview, { sliceIdx: 2, showTumor: true, seed: 42 });

  const steps = document.getElementById('pipeline-steps');
  steps.style.display = 'block';

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

  const complete = document.getElementById('pipeline-complete');
  complete.style.display = 'block';
  complete.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetPipeline() {
  running = false;
  stopSegmentationPlayback();
  segmentationViewerState.slice = 96;
  segmentationViewerState.plane = 'axial';

  document.getElementById('upload-zone').style.display = 'block';
  document.getElementById('load-btn').disabled = false;
  document.getElementById('scan-loaded').style.display = 'none';
  document.getElementById('pipeline-steps').style.display = 'none';
  document.getElementById('pipeline-complete').style.display = 'none';

  for (let stepNum = 0; stepNum <= 4; stepNum++) {
    const step = document.getElementById(`step-${stepNum}`);
    if (step) step.classList.remove('active', 'done', 'skip');

    const status = document.getElementById(`status-${stepNum}`);
    if (status) {
      status.textContent = 'Waiting';
      status.className = 'p-status';
    }

    const body = document.getElementById(`body-${stepNum}`);
    if (body) body.style.display = 'none';

    const fill = document.getElementById(`fill-${stepNum}`);
    if (fill) fill.style.width = '0%';

    const label = document.getElementById(`label-${stepNum}`);
    if (label) label.textContent = 'Initialising...';

    const progress = document.getElementById(`prog-${stepNum}`);
    if (progress) progress.style.display = 'block';

    const output = document.getElementById(`out-${stepNum}`);
    if (output) output.style.display = 'none';
  }

  ['slice-row', 'vote-table', 'xai-grid', 'report-text'].forEach(id => {
    const element = document.getElementById(id);
    if (element) element.innerHTML = '';
  });

  const confBar = document.getElementById('conf-bar');
  if (confBar) confBar.style.width = '0%';

  renderStaticPreviews();
}

window.addEventListener('scroll', () => {
  document.getElementById('header').style.boxShadow =
    window.scrollY > 8 ? '0 4px 24px rgba(0,0,0,0.45)' : 'none';
});

renderStaticPreviews();
