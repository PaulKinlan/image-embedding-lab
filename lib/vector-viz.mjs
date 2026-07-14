// Pure vector→image core for the Vector Visualizer (vector-viz.html).
//
// Same contract as lib/experiment.mjs: no DOM, no model runtime — just math — so the browser
// page and Node tests share one implementation and cannot drift. Every renderer returns
// { width, height, pixels: Uint8ClampedArray (RGBA), map: Int32Array } where map[y*w+x] is the
// vector dimension that produced that pixel (-1 = padding), so the UI can show "dim 217 = 0.043"
// on hover.

// --- Vector utilities ---

export function cosineSim(a, b) {
  let dot = 0, ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; ma += a[i] * a[i]; mb += b[i] * b[i]; }
  return dot / (Math.sqrt(ma) * Math.sqrt(mb) || 1);
}

export function l2normalize(v) {
  let mag = 0;
  for (let i = 0; i < v.length; i++) mag += v[i] * v[i];
  mag = Math.sqrt(mag) || 1;
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / mag;
  return out;
}

/** Spherical interpolation between two (normalized) vectors; endpoints exact. */
export function slerp(a, b, t) {
  const an = l2normalize(a), bn = l2normalize(b);
  let dot = 0;
  for (let i = 0; i < an.length; i++) dot += an[i] * bn[i];
  dot = Math.min(1, Math.max(-1, dot));
  const theta = Math.acos(dot);
  if (theta < 1e-6) return an;
  const s = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / s, wb = Math.sin(t * theta) / s;
  const out = new Float64Array(an.length);
  for (let i = 0; i < an.length; i++) out[i] = wa * an[i] + wb * bn[i];
  return out;
}

/** Element-wise difference a-b (for the diff view). */
export function diffVector(a, b) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - (b[i] ?? 0);
  return out;
}

// --- Normalization: map raw vector values → [0,1] for display ---
// 'symmetric' keeps zero at 0.5 (best for signed embeddings and diffs); 'minmax' uses the full
// display range; 'zscore' squashes outliers through a sigmoid; 'clip' assumes roughly ±0.5.

export const NORMALIZATIONS = ["symmetric", "minmax", "zscore", "clip"];

export function normalizeVector(v, mode = "symmetric") {
  const n = v.length;
  const out = new Float64Array(n);
  if (mode === "minmax") {
    let lo = Infinity, hi = -Infinity;
    for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; }
    const span = hi - lo || 1;
    for (let i = 0; i < n; i++) out[i] = (v[i] - lo) / span;
  } else if (mode === "zscore") {
    let mean = 0;
    for (const x of v) mean += x;
    mean /= n;
    let sd = 0;
    for (const x of v) sd += (x - mean) ** 2;
    sd = Math.sqrt(sd / n) || 1;
    for (let i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(-(v[i] - mean) / sd));
  } else if (mode === "clip") {
    for (let i = 0; i < n; i++) out[i] = Math.min(1, Math.max(0, v[i] + 0.5));
  } else { // symmetric (default)
    let maxAbs = 0;
    for (const x of v) { const a = Math.abs(x); if (a > maxAbs) maxAbs = a; }
    maxAbs = maxAbs || 1;
    for (let i = 0; i < n; i++) out[i] = 0.5 + v[i] / (2 * maxAbs);
  }
  return out;
}

// --- Ordering: permute dimensions before display ---
// Embedding dimension order is arbitrary, so re-ordering is a legitimate lens: sorted shows the
// value distribution; shuffled(seed) breaks any accidental training-order structure but stays
// stable across vectors so two images remain comparable.

export const ORDERINGS = ["natural", "sorted", "magnitude", "shuffled"];

export function orderIndices(v, mode = "natural", seed = 42) {
  const idx = Array.from({ length: v.length }, (_, i) => i);
  if (mode === "sorted") idx.sort((a, b) => v[b] - v[a]);
  else if (mode === "magnitude") idx.sort((a, b) => Math.abs(v[b]) - Math.abs(v[a]));
  else if (mode === "shuffled") {
    const rand = mulberry32(seed);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
  }
  return idx;
}

export function applyOrder(v, idx) {
  const out = new Float64Array(idx.length);
  for (let i = 0; i < idx.length; i++) out[i] = v[idx[i]];
  return out;
}

// --- Seeded PRNG + gaussian (deterministic: same vector → same image) ---

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianRand(rand) {
  let u = 0, w = 0;
  while (u === 0) u = rand();
  while (w === 0) w = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
}

// --- Colormaps: t∈[0,1] → [r,g,b] ---
// Anchor-interpolated approximations of the matplotlib maps (visually faithful, tiny).

const CM_ANCHORS = {
  viridis: [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]],
  plasma: [[13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137], [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33]],
  magma: [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191]],
  coolwarm: [[59, 76, 192], [124, 159, 249], [192, 212, 245], [242, 242, 242], [245, 212, 187], [238, 132, 104], [180, 4, 38]],
};

function anchorLookup(anchors, t) {
  const x = Math.min(1, Math.max(0, t)) * (anchors.length - 1);
  const i = Math.min(anchors.length - 2, Math.floor(x));
  const f = x - i;
  const a = anchors[i], b = anchors[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export const COLORMAPS = {
  gray: (t) => { const v = Math.round(255 * Math.min(1, Math.max(0, t))); return [v, v, v]; },
  viridis: (t) => anchorLookup(CM_ANCHORS.viridis, t),
  plasma: (t) => anchorLookup(CM_ANCHORS.plasma, t),
  magma: (t) => anchorLookup(CM_ANCHORS.magma, t),
  coolwarm: (t) => anchorLookup(CM_ANCHORS.coolwarm, t),
};

export function hslToRgb(h, s, l) {
  h = ((h % 360) + 360) % 360;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

// --- Hilbert curve (locality-preserving 1D→2D) ---

export function nextPow2(n) {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

/** Map distance d along the Hilbert curve to (x,y) on a side×side grid (side = power of 2). */
export function hilbertD2XY(side, d) {
  let rx, ry, x = 0, y = 0, t = d;
  for (let s = 1; s < side; s *= 2) {
    rx = 1 & Math.floor(t / 2);
    ry = 1 & (t ^ rx);
    if (ry === 0) {
      if (rx === 1) { x = s - 1 - x; y = s - 1 - y; }
      const tmp = x; x = y; y = tmp;
    }
    x += s * rx;
    y += s * ry;
    t = Math.floor(t / 4);
  }
  return [x, y];
}

// --- Renderers ---

function makeBuffer(width, height) {
  // Pre-fill with a faint neutral gray so cells no dimension reaches (grid padding) read as
  // deliberate padding rather than a rendering hole, in both light and dark themes.
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < pixels.length; i += 4) {
    pixels[i] = 127; pixels[i + 1] = 127; pixels[i + 2] = 127; pixels[i + 3] = 70;
  }
  const map = new Int32Array(width * height).fill(-1);
  return { width, height, pixels, map };
}

function putPx(buf, x, y, rgb, dim) {
  const o = (y * buf.width + x) * 4;
  buf.pixels[o] = rgb[0]; buf.pixels[o + 1] = rgb[1]; buf.pixels[o + 2] = rgb[2]; buf.pixels[o + 3] = 255;
  if (dim != null) buf.map[y * buf.width + x] = dim;
}

/** Row-major grid. rgb: consume dims 3-at-a-time as R,G,B; else one dim per pixel via colormap. */
export function renderGrid(values01, { rgb = false, colormap = COLORMAPS.viridis, dimIndex = null } = {}) {
  const cells = rgb ? Math.ceil(values01.length / 3) : values01.length;
  const width = Math.ceil(Math.sqrt(cells));
  const height = Math.ceil(cells / width);
  const buf = makeBuffer(width, height);
  for (let c = 0; c < cells; c++) {
    const x = c % width, y = Math.floor(c / width);
    if (rgb) {
      const r = values01[c * 3] ?? 0.5, g = values01[c * 3 + 1] ?? 0.5, b = values01[c * 3 + 2] ?? 0.5;
      putPx(buf, x, y, [r * 255, g * 255, b * 255], dimIndex ? dimIndex[c * 3] : c * 3);
    } else {
      putPx(buf, x, y, colormap(values01[c]), dimIndex ? dimIndex[c] : c);
    }
  }
  return buf;
}

/** 1×N barcode strip (UI stretches it vertically). */
export function renderBarcode(values01, { colormap = COLORMAPS.viridis, dimIndex = null } = {}) {
  const buf = makeBuffer(values01.length, 1);
  for (let i = 0; i < values01.length; i++) {
    putPx(buf, i, 0, colormap(values01[i]), dimIndex ? dimIndex[i] : i);
  }
  return buf;
}

/** Hilbert-curve walk: consecutive dims stay spatial neighbours. */
export function renderHilbert(values01, { rgb = false, colormap = COLORMAPS.viridis, dimIndex = null } = {}) {
  const cells = rgb ? Math.ceil(values01.length / 3) : values01.length;
  const side = nextPow2(Math.ceil(Math.sqrt(cells)));
  const buf = makeBuffer(side, side);
  for (let c = 0; c < cells; c++) {
    const [x, y] = hilbertD2XY(side, c);
    if (rgb) {
      const r = values01[c * 3] ?? 0.5, g = values01[c * 3 + 1] ?? 0.5, b = values01[c * 3 + 2] ?? 0.5;
      putPx(buf, x, y, [r * 255, g * 255, b * 255], dimIndex ? dimIndex[c * 3] : c * 3);
    } else {
      putPx(buf, x, y, colormap(values01[c]), dimIndex ? dimIndex[c] : c);
    }
  }
  return buf;
}

/**
 * Self-similarity matrix: pixel (i,j) = v[i]*v[j], symmetrically normalized so 0 → colormap
 * midpoint. Use a diverging map (coolwarm). map[] holds i (row dim); UI derives j from x.
 */
export function renderSimMatrix(v, { colormap = COLORMAPS.coolwarm } = {}) {
  const n = v.length;
  const buf = makeBuffer(n, n);
  // Scale by a high percentile of |vᵢ·vⱼ|, not the max: for an L2-normalized vector the max
  // product (one outlier squared) dwarfs the off-diagonal mass, which would push every other
  // pixel to the colormap midpoint and render the matrix as a blank wash. |vᵢ·vⱼ|=|vᵢ||vⱼ|, so
  // the product percentile comes straight from the sorted |v| percentile, no n² pass needed.
  const absSorted = Array.from(v, Math.abs).sort((a, b) => a - b);
  const p90 = absSorted[Math.floor(0.9 * (n - 1))];
  const scale = p90 * p90 || 1;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const t = Math.min(1, Math.max(0, 0.5 + (v[i] * v[j]) / (2 * scale)));
      putPx(buf, j, i, colormap(t), i);
    }
  }
  return buf;
}

/** Consecutive pairs as complex numbers: hue = phase, lightness = magnitude vs the max pair. */
export function renderPhase(v, { dimIndex = null } = {}) {
  const pairs = Math.floor(v.length / 2);
  const width = Math.ceil(Math.sqrt(pairs));
  const height = Math.ceil(pairs / width);
  const buf = makeBuffer(width, height);
  let maxMag = 0;
  for (let p = 0; p < pairs; p++) {
    const m = Math.hypot(v[p * 2], v[p * 2 + 1]);
    if (m > maxMag) maxMag = m;
  }
  maxMag = maxMag || 1;
  for (let p = 0; p < pairs; p++) {
    const re = v[p * 2], im = v[p * 2 + 1];
    const hue = (Math.atan2(im, re) * 180) / Math.PI;
    const light = 0.15 + 0.7 * (Math.hypot(re, im) / maxMag);
    putPx(buf, p % width, Math.floor(p / width), hslToRgb(hue, 0.85, light), dimIndex ? dimIndex[p * 2] : p * 2);
  }
  return buf;
}

/**
 * Random-projection fingerprint: each output pixel is the dot product of the vector with a
 * seeded gaussian direction — a smooth, unique "texture" for the whole vector. Same seed +
 * same vector → identical image; nearby vectors → visibly similar textures.
 */
export function renderProjection(v, { size = 64, seed = 1234, colormap = COLORMAPS.magma } = {}) {
  const buf = makeBuffer(size, size);
  const rand = mulberry32(seed);
  const vals = new Float64Array(size * size);
  let maxAbs = 0;
  for (let p = 0; p < size * size; p++) {
    let dot = 0;
    for (let i = 0; i < v.length; i++) dot += v[i] * gaussianRand(rand);
    vals[p] = dot;
    const a = Math.abs(dot);
    if (a > maxAbs) maxAbs = a;
  }
  maxAbs = maxAbs || 1;
  for (let p = 0; p < size * size; p++) {
    putPx(buf, p % size, Math.floor(p / size), colormap(0.5 + vals[p] / (2 * maxAbs)), -1);
  }
  return buf;
}

/**
 * Radial "flower": one spoke per dimension, radius = normalized value. Returns polygon points
 * in [-1,1] unit space for the UI to stroke/fill on a vector canvas.
 */
export function flowerPoints(values01) {
  const n = values01.length;
  const pts = [];
  for (let i = 0; i < n; i++) {
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
    const r = 0.08 + 0.9 * values01[i];
    pts.push([Math.cos(angle) * r, Math.sin(angle) * r]);
  }
  return pts;
}

/**
 * Normalize a SET of vectors jointly (same params across all of them), so cross-vector
 * differences stay visible. Per-vector normalization would rescale each row independently and
 * hide exactly the shifts a rotation sweep is trying to show.
 */
export function normalizeJoint(vectors, mode = "zscore") {
  const all = [];
  for (const v of vectors) for (const x of v) all.push(x);
  const n = all.length;
  let map;
  if (mode === "minmax") {
    let lo = Infinity, hi = -Infinity;
    for (const x of all) { if (x < lo) lo = x; if (x > hi) hi = x; }
    const span = hi - lo || 1;
    map = (x) => (x - lo) / span;
  } else if (mode === "zscore") {
    let mean = 0;
    for (const x of all) mean += x;
    mean /= n;
    let sd = 0;
    for (const x of all) sd += (x - mean) ** 2;
    sd = Math.sqrt(sd / n) || 1;
    map = (x) => 1 / (1 + Math.exp(-(x - mean) / sd));
  } else if (mode === "clip") {
    map = (x) => Math.min(1, Math.max(0, x + 0.5));
  } else { // symmetric
    let maxAbs = 0;
    for (const x of all) { const a = Math.abs(x); if (a > maxAbs) maxAbs = a; }
    maxAbs = maxAbs || 1;
    map = (x) => 0.5 + x / (2 * maxAbs);
  }
  return vectors.map((v) => Float64Array.from(v, map));
}

/**
 * Stack of barcodes: one row per vector (e.g. per rotation angle), one column per dimension.
 * Pass ALREADY jointly-normalized rows (see normalizeJoint). map[] holds the dimension index,
 * same for every row; the UI derives the row (angle) from y.
 */
export function renderStack(rows01, { colormap = COLORMAPS.viridis, dimIndex = null } = {}) {
  const width = rows01[0].length, height = rows01.length;
  const buf = makeBuffer(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      putPx(buf, x, y, colormap(rows01[y][x]), dimIndex ? dimIndex[x] : x);
    }
  }
  return buf;
}

/** Square-spiral walk: dim 0 at the centre, later dims spiral outward. */
export function spiralCoords(cells) {
  const side = Math.ceil(Math.sqrt(cells));
  const cx = Math.floor((side - 1) / 2), cy = Math.floor((side - 1) / 2);
  const coords = [];
  let x = cx, y = cy, dx = 1, dy = 0, run = 1, steps = 0, turns = 0;
  while (coords.length < cells) {
    if (x >= 0 && x < side && y >= 0 && y < side) coords.push([x, y]);
    x += dx; y += dy; steps++;
    if (steps === run) {
      steps = 0;
      const t = dx; dx = -dy; dy = t;   // turn left: (1,0)→(0,-1)? keep consistent CW/CCW
      turns++;
      if (turns % 2 === 0) run++;
    }
  }
  return { side, coords };
}

/** Spiral walk render: like the Hilbert view but winding outward from the centre. */
export function renderSpiral(values01, { rgb = false, colormap = COLORMAPS.viridis, dimIndex = null } = {}) {
  const cells = rgb ? Math.ceil(values01.length / 3) : values01.length;
  const { side, coords } = spiralCoords(cells);
  const buf = makeBuffer(side, side);
  for (let c = 0; c < cells; c++) {
    const [x, y] = coords[c];
    if (rgb) {
      const r = values01[c * 3] ?? 0.5, g = values01[c * 3 + 1] ?? 0.5, b = values01[c * 3 + 2] ?? 0.5;
      putPx(buf, x, y, [r * 255, g * 255, b * 255], dimIndex ? dimIndex[c * 3] : c * 3);
    } else {
      putPx(buf, x, y, colormap(values01[c]), dimIndex ? dimIndex[c] : c);
    }
  }
  return buf;
}

/** Recurrence plot: pixel (i,j) = |vᵢ − vⱼ|, scaled by its p90 (same reasoning as simmatrix). */
export function renderRecurrence(v, { colormap = COLORMAPS.magma } = {}) {
  const n = v.length;
  const buf = makeBuffer(n, n);
  const diffs = [];
  const stride = Math.max(1, Math.floor(n / 96));   // sample pairs for the percentile
  for (let i = 0; i < n; i += stride) for (let j = 0; j < n; j += stride) diffs.push(Math.abs(v[i] - v[j]));
  diffs.sort((a, b) => a - b);
  const p90 = diffs[Math.floor(0.9 * (diffs.length - 1))] || 1;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      putPx(buf, j, i, colormap(Math.min(1, Math.abs(v[i] - v[j]) / p90)), i);
    }
  }
  return buf;
}

/** Naive DFT magnitude spectrum (k = 0…n/2), normalized by the max non-DC bin. */
export function dftMagnitudes(v) {
  const n = v.length, half = Math.floor(n / 2) + 1;
  const mags = new Float64Array(half);
  for (let k = 0; k < half; k++) {
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) {
      const a = (-2 * Math.PI * k * i) / n;
      re += v[i] * Math.cos(a);
      im += v[i] * Math.sin(a);
    }
    mags[k] = Math.hypot(re, im);
  }
  return mags;
}

/** Frequency spectrum as a bar image: x = frequency bin, bar height = magnitude. */
export function renderSpectrum(v, { colormap = COLORMAPS.plasma, height = 96 } = {}) {
  const mags = dftMagnitudes(v);
  let max = 0;
  for (let k = 1; k < mags.length; k++) if (mags[k] > max) max = mags[k];
  max = max || 1;
  const buf = makeBuffer(mags.length, height);
  for (let x = 0; x < mags.length; x++) {
    const t = Math.min(1, mags[x] / max);
    const barH = Math.round(t * height);
    for (let y = 0; y < height; y++) {
      if (height - 1 - y < barH) putPx(buf, x, y, colormap(t), x === 0 ? -1 : x);
    }
  }
  return buf;
}

/** Waveform polyline points in [-1,1] unit space (x = dim, y = value; 0 at the midline). */
export function waveformPoints(values01) {
  const n = values01.length;
  const pts = [];
  for (let i = 0; i < n; i++) {
    pts.push([n === 1 ? 0 : (i / (n - 1)) * 2 - 1, -(values01[i] * 2 - 1)]);
  }
  return pts;
}

/**
 * Interference texture: every dimension contributes one plane wave — orientation from the
 * golden angle, frequency from the dim index, amplitude from the value. The sum is a smooth,
 * unique texture; similar vectors → visibly similar interference patterns. Deterministic.
 */
export function renderInterference(v, { size = 96, colormap = COLORMAPS.magma } = {}) {
  const n = v.length;
  const golden = Math.PI * (3 - Math.sqrt(5));
  const th = new Float64Array(n), fx = new Float64Array(n), fy = new Float64Array(n);
  for (let d = 0; d < n; d++) {
    th[d] = d * golden;
    const f = (2 * Math.PI * (1 + (d % 13))) / size;
    fx[d] = f * Math.cos(th[d]);
    fy[d] = f * Math.sin(th[d]);
  }
  const vals = new Float64Array(size * size);
  let maxAbs = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let s = 0;
      for (let d = 0; d < n; d++) s += v[d] * Math.cos(fx[d] * x + fy[d] * y + th[d]);
      vals[y * size + x] = s;
      const a = Math.abs(s);
      if (a > maxAbs) maxAbs = a;
    }
  }
  maxAbs = maxAbs || 1;
  const buf = makeBuffer(size, size);
  for (let p = 0; p < size * size; p++) {
    putPx(buf, p % size, Math.floor(p / size), colormap(0.5 + vals[p] / (2 * maxAbs)), -1);
  }
  return buf;
}

export const MODES = [
  { key: "rgb-grid", label: "RGB triples → grid", desc: "Consecutive triples of dimensions become one pixel's R,G,B" },
  { key: "gray-grid", label: "Heatmap grid", desc: "One dimension per pixel through a colormap" },
  { key: "hilbert", label: "Hilbert curve", desc: "Locality-preserving walk — neighbouring dims stay neighbouring pixels" },
  { key: "hilbert-rgb", label: "Hilbert + RGB triples", desc: "Hilbert walk, triples as colour" },
  { key: "spiral", label: "Spiral walk", desc: "Dim 0 at the centre, later dimensions wind outward" },
  { key: "barcode", label: "Barcode", desc: "All dimensions as one 1-pixel-tall strip" },
  { key: "waveform", label: "Waveform", desc: "The vector as an audio-style waveform: x = dimension, y = value" },
  { key: "simmatrix", label: "Self-similarity matrix", desc: "Pixel (i,j) = vᵢ·vⱼ — the vector's outer product with itself" },
  { key: "recurrence", label: "Recurrence plot", desc: "Pixel (i,j) = |vᵢ − vⱼ| — distance structure between dimensions" },
  { key: "spectrum", label: "Frequency spectrum", desc: "DFT of the vector-as-signal: which 'wavelengths' across dimensions carry the energy" },
  { key: "phase", label: "Phase wheel", desc: "Dimension pairs as complex numbers: hue = angle, brightness = magnitude" },
  { key: "projection", label: "Random projection", desc: "Seeded gaussian projection to a 64×64 texture — a stable fingerprint" },
  { key: "interference", label: "Interference texture", desc: "Each dimension is a plane wave (golden-angle orientation); their sum is the vector's unique moiré" },
  { key: "flower", label: "Radial flower", desc: "One spoke per dimension, radius = value" },
];
