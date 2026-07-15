// Experiment 1 — PATCH SHUFFLE. Scramble each image's g×g tiles with a fixed seeded
// permutation: content preserved exactly, layout destroyed. A "bag of semantics" embedding
// barely moves; a layout-encoding embedding collapses toward the different-image floor.
// Thesis prediction: Florence (VLM) and DINOv2 drop hardest, CLIP least.
import fs from "node:fs";
import { MODELS, CORPUS, embed, floor, cosineSim, meanStd, fmt, normalizedSim } from "./lib-harness.mjs";

const CONDITIONS = [
  { name: "shuffle 2×2", p: { shuffle: 2 } },
  { name: "shuffle 4×4", p: { shuffle: 4 } },
  { name: "shuffle 8×8", p: { shuffle: 8 } },
];

const results = {};
for (const model of Object.keys(MODELS)) {
  console.log(`\n=== ${model} (${MODELS[model].id}) ===`);
  const fl = await floor(model);
  const rows = {};
  for (const cond of CONDITIONS) {
    const sims = [];
    for (const file of CORPUS) {
      const base = (await embed(model, file)).pooled;
      const t = (await embed(model, file, cond.p)).pooled;
      sims.push(cosineSim(base, t));
    }
    const { mean, std } = meanStd(sims);
    rows[cond.name] = { mean, std, normalized: normalizedSim(mean, fl.mean) };
    console.log(`${cond.name}  sim ${fmt(mean)} ±${fmt(std)}  floor-normalized ${fmt(normalizedSim(mean, fl.mean))}`);
  }
  results[model] = { floor: fl.mean, conditions: rows };
  console.log(`floor ${fmt(fl.mean)}`);
}
fs.writeFileSync("results-patch-shuffle.json", JSON.stringify({ date: process.env.RUN_DATE || null, corpus: CORPUS.length, results }, null, 2));
console.log("\nwritten results-patch-shuffle.json");
