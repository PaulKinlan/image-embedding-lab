// Experiment 8 — SAME-IMAGE THRESHOLDS. At what cosine can you say "these are the same image"
// with high confidence? For each model: the impostor distribution (all 105 different-image
// pairs, identity renders) vs same-image similarities under transform groups (rotations,
// crops, flips, shuffles). We report the impostor ceiling (max different-image sim — above it,
// zero false positives ON THIS CORPUS) and recall at that threshold per group, and export the
// raw distributions for the explainer's interactive widget.
import fs from "node:fs";
import { MODELS, CORPUS, embed, cosineSim, fmt } from "./lib-harness.mjs";

const GROUPS = {
  rotations: [{ rotation: 90 }, { rotation: 180 }, { rotation: 270 }, { rotation: 45 }],
  crops: [{ crop: 0.10 }, { crop: 0.20 }, { crop: 0.35 }],
  flips: [{ flip: "horizontal" }],
  shuffles: [{ shuffle: 4 }, { shuffle: 8 }],
};

const out = {};
for (const model of Object.keys(MODELS)) {
  console.log(`\n=== ${model} ===`);
  const idVecs = {};
  for (const f of CORPUS) idVecs[f] = (await embed(model, f)).pooled;
  const impostor = [];
  for (let i = 0; i < CORPUS.length; i++) {
    for (let j = i + 1; j < CORPUS.length; j++) impostor.push(cosineSim(idVecs[CORPUS[i]], idVecs[CORPUS[j]]));
  }
  const ceiling = Math.max(...impostor);
  const groups = {};
  for (const [name, ps] of Object.entries(GROUPS)) {
    const sims = [];
    for (const f of CORPUS) {
      for (const p of ps) sims.push(cosineSim(idVecs[f], (await embed(model, f, p)).pooled));
    }
    const recallAtCeiling = sims.filter((s) => s > ceiling).length / sims.length;
    groups[name] = { sims, recallAtCeiling };
    console.log(`${name.padEnd(10)} n=${sims.length}  min ${fmt(Math.min(...sims))}  recall@ceiling ${fmt(recallAtCeiling, 2)}`);
  }
  out[model] = { impostor, ceiling, groups };
  console.log(`impostor ceiling ${fmt(ceiling)} (n=${impostor.length} pairs)`);
}
fs.writeFileSync("results-same-image.json", JSON.stringify({ corpus: CORPUS.length, results: out }, null, 2));
console.log("\nwritten results-same-image.json");
