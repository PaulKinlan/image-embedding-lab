// Node sanity tests for lib/vector-viz.mjs. Run: node scripts/test-vector-viz.mjs
import {
  cosineSim, l2normalize, slerp, diffVector, normalizeVector, NORMALIZATIONS,
  orderIndices, applyOrder, ORDERINGS, hilbertD2XY, nextPow2,
  renderGrid, renderBarcode, renderHilbert, renderSimMatrix, renderPhase, renderProjection,
  flowerPoints, COLORMAPS, MODES, normalizeJoint, renderStack,
  spiralCoords, renderSpiral, renderRecurrence, dftMagnitudes, renderSpectrum,
  waveformPoints, renderInterference,
} from "../lib/vector-viz.mjs";

let failures = 0;
function check(name, cond) {
  if (!cond) { failures++; console.error(`FAIL: ${name}`); }
  else console.log(`ok: ${name}`);
}

// Deterministic pseudo-embedding (values roughly ±0.1, like a real normalized embedding)
function fakeVec(n, seed = 7) {
  const v = new Float64Array(n);
  let s = seed;
  for (let i = 0; i < n; i++) {
    s = (s * 16807) % 2147483647;
    v[i] = (s / 2147483647 - 0.5) * 0.2;
  }
  return l2normalize(v);
}

// Hilbert: bijective over the full grid for every side we'll hit (512→23→32, 768→28→32)
for (const side of [8, 16, 32]) {
  const seen = new Set();
  for (let d = 0; d < side * side; d++) {
    const [x, y] = hilbertD2XY(side, d);
    check(`hilbert(${side}) in-bounds d=${d}`, x >= 0 && x < side && y >= 0 && y < side) || null;
    seen.add(y * side + x);
    if (x < 0 || x >= side || y < 0 || y >= side) break;
  }
  check(`hilbert(${side}) bijective (${seen.size}/${side * side})`, seen.size === side * side);
  // adjacency: consecutive d are 4-neighbours (the locality property we're claiming in the UI)
  let adjacent = true;
  for (let d = 1; d < side * side; d++) {
    const [x1, y1] = hilbertD2XY(side, d - 1);
    const [x2, y2] = hilbertD2XY(side, d);
    if (Math.abs(x1 - x2) + Math.abs(y1 - y2) !== 1) { adjacent = false; break; }
  }
  check(`hilbert(${side}) consecutive cells adjacent`, adjacent);
}

// Normalization: all modes land in [0,1]; symmetric maps 0 → 0.5
const v512 = fakeVec(512);
for (const mode of NORMALIZATIONS) {
  const nv = normalizeVector(v512, mode);
  let lo = Infinity, hi = -Infinity;
  for (const x of nv) { if (x < lo) lo = x; if (x > hi) hi = x; }
  check(`normalize ${mode} in [0,1] (got ${lo.toFixed(3)}..${hi.toFixed(3)})`, lo >= 0 && hi <= 1);
}
const zeroed = new Float64Array(8); zeroed[0] = 0.5; zeroed[1] = -0.5;
const sym = normalizeVector(zeroed, "symmetric");
check("symmetric keeps 0 at 0.5", Math.abs(sym[2] - 0.5) < 1e-9 && Math.abs(sym[0] - 1) < 1e-9 && Math.abs(sym[1]) < 1e-9);

// Ordering: every mode is a valid permutation; shuffled is seed-stable
for (const mode of ORDERINGS) {
  const idx = orderIndices(v512, mode, 42);
  check(`order ${mode} is a permutation`, new Set(idx).size === 512);
}
check("shuffled seed-stable", JSON.stringify(orderIndices(v512, "shuffled", 42)) === JSON.stringify(orderIndices(v512, "shuffled", 42)));
const sortedVals = applyOrder(v512, orderIndices(v512, "sorted"));
check("sorted descending", sortedVals.every((x, i) => i === 0 || sortedVals[i - 1] >= x));

// Renderers: expected geometry for CLIP(512) / SigLIP(768) / DINOv2(384) dims
for (const n of [384, 512, 768]) {
  const vec = fakeVec(n);
  const nv = normalizeVector(vec, "symmetric");
  const grid = renderGrid(nv);
  check(`gray-grid(${n}) fits all dims`, grid.width * grid.height >= n && grid.pixels.length === grid.width * grid.height * 4);
  const rgbG = renderGrid(nv, { rgb: true });
  check(`rgb-grid(${n}) has ceil(n/3) cells area`, rgbG.width * rgbG.height >= Math.ceil(n / 3));
  const hil = renderHilbert(nv);
  check(`hilbert(${n}) square pow2 side`, hil.width === hil.height && hil.width === nextPow2(Math.ceil(Math.sqrt(n))));
  const bar = renderBarcode(nv);
  check(`barcode(${n}) is n×1`, bar.width === n && bar.height === 1);
  const sim = renderSimMatrix(vec);
  check(`simmatrix(${n}) is n×n`, sim.width === n && sim.height === n);
  const ph = renderPhase(vec);
  check(`phase(${n}) covers n/2 pairs`, ph.width * ph.height >= Math.floor(n / 2));
  const fl = flowerPoints(nv);
  check(`flower(${n}) n points in unit box`, fl.length === n && fl.every(([x, y]) => Math.abs(x) <= 1 && Math.abs(y) <= 1));
}

// map[] hover support: grid pixel 0 is dim 0, barcode pixel i is dim i
const g = renderGrid(normalizeVector(v512), {});
check("grid map[0] = 0", g.map[0] === 0);
const b = renderBarcode(normalizeVector(v512));
check("barcode map[10] = 10", b.map[10] === 10);

// Projection: deterministic and sensitive to the vector
const p1 = renderProjection(v512, { size: 16, seed: 9 });
const p2 = renderProjection(v512, { size: 16, seed: 9 });
const p3 = renderProjection(fakeVec(512, 99), { size: 16, seed: 9 });
check("projection deterministic", p1.pixels.every((x, i) => x === p2.pixels[i]));
check("projection vector-sensitive", !p1.pixels.every((x, i) => x === p3.pixels[i]));

// slerp: endpoints exact, midpoint between
const a = fakeVec(512, 1), c = fakeVec(512, 2);
check("slerp t=0 → a", cosineSim(slerp(a, c, 0), a) > 0.999999);
check("slerp t=1 → b", cosineSim(slerp(a, c, 1), c) > 0.999999);
const mid = slerp(a, c, 0.5);
check("slerp midpoint equidistant", Math.abs(cosineSim(mid, a) - cosineSim(mid, c)) < 1e-6);

// diff + colormaps
const d = diffVector(a, c);
check("diff correct", Math.abs(d[0] - (a[0] - c[0])) < 1e-12);
for (const [name, fn] of Object.entries(COLORMAPS)) {
  const [r, gg, bb] = fn(0.5);
  check(`colormap ${name} valid rgb`, [r, gg, bb].every((x) => x >= 0 && x <= 255));
}
check(`MODES defined (${MODES.length})`, MODES.length === 14);

// Spiral: visits `cells` distinct in-bounds positions for every dim count we use
for (const n of [171, 384, 512, 768]) {
  const { side, coords } = spiralCoords(n);
  const seen = new Set(coords.map(([x, y]) => y * side + x));
  check(`spiral(${n}) ${coords.length} unique in-bounds cells`,
    coords.length === n && seen.size === n && coords.every(([x, y]) => x >= 0 && x < side && y >= 0 && y < side));
}
{
  const nv = normalizeVector(fakeVec(512), "zscore");
  const sp = renderSpiral(nv);
  check("renderSpiral square", sp.width === sp.height && sp.width === Math.ceil(Math.sqrt(512)));
  check("renderSpiral map centre = dim 0", sp.map[Math.floor((sp.width - 1) / 2) * sp.width + Math.floor((sp.width - 1) / 2)] === 0);
}

// Recurrence: n×n, diagonal = zero distance = colormap(0)
{
  const v = fakeVec(96);
  const rc = renderRecurrence(v, { colormap: COLORMAPS.gray });
  check("recurrence n×n", rc.width === 96 && rc.height === 96);
  const o = (5 * 96 + 5) * 4;
  check("recurrence diagonal is 0-distance (black in gray cmap)", rc.pixels[o] === 0 && rc.pixels[o + 3] === 255);
}

// DFT: a pure cosine concentrates energy in its own bin
{
  const n = 128, freq = 7;
  const sig = new Float64Array(n);
  for (let i = 0; i < n; i++) sig[i] = Math.cos((2 * Math.PI * freq * i) / n);
  const mags = dftMagnitudes(sig);
  let argmax = 1;
  for (let k = 1; k < mags.length; k++) if (mags[k] > mags[argmax]) argmax = k;
  check(`dft pure tone lands in bin ${freq}`, argmax === freq);
  const sp = renderSpectrum(sig);
  check("spectrum geometry", sp.width === Math.floor(n / 2) + 1 && sp.height === 96);
}

// Waveform: n points spanning x∈[-1,1], y bounded
{
  const wf = waveformPoints(normalizeVector(fakeVec(512), "minmax"));
  check("waveform n points, x spans [-1,1]", wf.length === 512 && wf[0][0] === -1 && wf[511][0] === 1
    && wf.every(([x, y]) => Math.abs(y) <= 1));
}

// Interference: deterministic + vector-sensitive
{
  const i1 = renderInterference(fakeVec(384), { size: 24 });
  const i2 = renderInterference(fakeVec(384), { size: 24 });
  const i3 = renderInterference(fakeVec(384, 55), { size: 24 });
  check("interference deterministic", i1.pixels.every((x, i) => x === i2.pixels[i]));
  check("interference vector-sensitive", !i1.pixels.every((x, i) => x === i3.pixels[i]));
}

// Joint normalization: shared params across rows (a constant offset between rows must survive)
{
  const base = fakeVec(64, 3);
  const shifted = Float64Array.from(base, (x) => x + 0.05);
  for (const mode of NORMALIZATIONS) {
    const [r1, r2] = normalizeJoint([base, shifted], mode);
    let inBounds = true, ordered = true;
    for (let i = 0; i < 64; i++) {
      if (r1[i] < 0 || r1[i] > 1 || r2[i] < 0 || r2[i] > 1) inBounds = false;
      if (r2[i] < r1[i] - 1e-9) ordered = false;   // +offset must stay >= after joint mapping
    }
    check(`normalizeJoint ${mode} in-bounds & offset preserved`, inBounds && ordered);
  }
  const [p1] = normalizeJoint([base], "zscore");
  const [j1] = normalizeJoint([base, shifted], "zscore");
  check("normalizeJoint params depend on the whole set", p1.some((x, i) => Math.abs(x - j1[i]) > 1e-9));
}

// Stack renderer: rows × dims geometry, row y = vector y, map[] = dim index
{
  const rows = normalizeJoint([fakeVec(96, 1), fakeVec(96, 2), fakeVec(96, 3)], "zscore");
  const st = renderStack(rows);
  check("stack is dims wide × rows tall", st.width === 96 && st.height === 3);
  check("stack map is dim index", st.map[1] === 1 && st.map[96 + 5] === 5);
  const [r, g, b2] = COLORMAPS.viridis(rows[1][0]);
  const o = (1 * 96 + 0) * 4;
  check("stack row 1 col 0 matches colormap", st.pixels[o] === Math.round(r) || Math.abs(st.pixels[o] - r) <= 1);
}

console.log(failures ? `\n${failures} FAILURES` : "\nAll tests passed");
process.exit(failures ? 1 : 0);
