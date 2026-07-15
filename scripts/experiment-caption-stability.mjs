// Experiment 5 — CAPTION STABILITY UNDER ROTATION. Florence's EMBEDDING moves under rotation
// (~0.8 cosine at 90°) — but does what the model SAYS change? For each image and each angle,
// generate a <CAPTION> and compare it to the upright caption (exact match + token Jaccard).
// Encoder-moves + decoder-shrugs ⇒ the layout information is present but downstream-robust.
import fs from "node:fs";
import { CORPUS, caption, embed, cosineSim, meanStd, fmt } from "./lib-harness.mjs";

const ANGLES = [90, 180, 270];

const norm = (s) => s.toLowerCase().replace(/[^a-z0-9 ]/g, "").split(/\s+/).filter(Boolean);
function jaccard(a, b) {
  const A = new Set(norm(a)), B = new Set(norm(b));
  const inter = [...A].filter((x) => B.has(x)).length;
  return inter / (new Set([...A, ...B]).size || 1);
}

const perImage = [];
const stats = {};
for (const angle of ANGLES) stats[angle] = { jac: [], exact: 0, cos: [] };

for (const file of CORPUS) {
  const base = await caption(file);
  const baseVec = (await embed("florence", file)).pooled;
  const row = { file, base, angles: {} };
  for (const angle of ANGLES) {
    const c = await caption(file, { rotation: angle });
    const v = (await embed("florence", file, { rotation: angle })).pooled;
    const j = jaccard(base, c);
    const cs = cosineSim(baseVec, v);
    row.angles[angle] = { caption: c, jaccard: j, embeddingCos: cs };
    stats[angle].jac.push(j);
    stats[angle].cos.push(cs);
    if (j === 1) stats[angle].exact++;
  }
  perImage.push(row);
  console.log(`${file}: "${base}"`);
  for (const angle of ANGLES) console.log(`  ${angle}° (cos ${fmt(row.angles[angle].embeddingCos)}, jac ${fmt(row.angles[angle].jaccard, 2)}): "${row.angles[angle].caption}"`);
}

console.log("\n=== summary ===");
const summary = {};
for (const angle of ANGLES) {
  const j = meanStd(stats[angle].jac), c = meanStd(stats[angle].cos);
  summary[angle] = { meanJaccard: j.mean, exactRate: stats[angle].exact / CORPUS.length, meanEmbeddingCos: c.mean };
  console.log(`${angle}°  caption jaccard ${fmt(j.mean)}  exact ${stats[angle].exact}/${CORPUS.length}  embedding cos ${fmt(c.mean)}`);
}
fs.writeFileSync("results-caption-stability.json", JSON.stringify({ angles: ANGLES, summary, perImage }, null, 2));
console.log("\nwritten results-caption-stability.json");
