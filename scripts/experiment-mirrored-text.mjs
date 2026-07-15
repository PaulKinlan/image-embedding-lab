// Experiment 6 — MIRRORED TEXT. Horizontal flip preserves photographic semantics (a mirrored
// beach is still a beach) but destroys textual semantics (mirrored text is unreadable). If
// language-aligned encoders (CLIP/SigLIP) encode "meaning", flipping should hurt TEXT images
// far more than photos; DINOv2 (no language) should treat both alike. Florence, which reads
// text for OCR, should punish mirrored text hardest. Per-category floors keep it honest.
import fs from "node:fs";
import { MODELS, PHOTO_FILES, TEXTUAL_FILES, embed, floor, cosineSim, meanStd, fmt, normalizedSim } from "./lib-harness.mjs";

const CATEGORIES = { photos: PHOTO_FILES, textual: TEXTUAL_FILES };

const results = {};
for (const model of Object.keys(MODELS)) {
  console.log(`\n=== ${model} ===`);
  results[model] = {};
  for (const [cat, files] of Object.entries(CATEGORIES)) {
    const fl = await floor(model, files);
    const sims = [];
    for (const file of files) {
      const base = (await embed(model, file)).pooled;
      const flip = (await embed(model, file, { flip: "horizontal" })).pooled;
      sims.push(cosineSim(base, flip));
    }
    const { mean, std } = meanStd(sims);
    const norm = normalizedSim(mean, fl.mean);
    results[model][cat] = { flipSim: mean, std, floor: fl.mean, normalized: norm, n: files.length };
    console.log(`${cat.padEnd(8)} flip-H sim ${fmt(mean)} ±${fmt(std)}  floor ${fmt(fl.mean)}  normalized ${fmt(norm)}`);
  }
  const gap = results[model].photos.normalized - results[model].textual.normalized;
  results[model].photoMinusTextualNormalized = gap;
  console.log(`photo − textual (normalized): ${fmt(gap)}`);
}
fs.writeFileSync("results-mirrored-text.json", JSON.stringify({ categories: { photos: PHOTO_FILES.length, textual: TEXTUAL_FILES.length }, results }, null, 2));
console.log("\nwritten results-mirrored-text.json");
