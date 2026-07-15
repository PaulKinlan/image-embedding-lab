// Experiment 7b — CAPTION RETRIEVAL ON GEMINI EMBEDDING 2. Same protocol as the CLIP version:
// after each transform, does the image still retrieve its own caption (top-1 of 15) in the
// shared text↔image space? Gemini Embedding 2 is VLM-native (Gemini architecture), so this asks
// whether a production VLM-lineage embedding space keeps semantics under layout destruction.
import fs from "node:fs";
import { CORPUS, CAPTIONS, embed, cosineSim, fmt, geminiEmbedText } from "./lib-harness.mjs";

const CONDITIONS = [
  { name: "identity", p: {} },
  { name: "rotate 90°", p: { rotation: 90 } },
  { name: "rotate 180°", p: { rotation: 180 } },
  { name: "rotate 45°", p: { rotation: 45 } },
  { name: "crop 10%", p: { crop: 0.10 } },
  { name: "crop 20%", p: { crop: 0.20 } },
  { name: "crop 35%", p: { crop: 0.35 } },
  { name: "rot 90° + crop 10%", p: { rotation: 90, crop: 0.10 } },
  { name: "rot 90° + crop 20%", p: { rotation: 90, crop: 0.20 } },
  { name: "rot 90° + crop 35%", p: { rotation: 90, crop: 0.35 } },
  { name: "rot 45° + crop 20%", p: { rotation: 45, crop: 0.20 } },
  { name: "rot 180° + crop 20%", p: { rotation: 180, crop: 0.20 } },
  { name: "shuffle 4×4", p: { shuffle: 4 } },
  { name: "flip H", p: { flip: "horizontal" } },
  { name: "blur", p: { blur: 3 } },
];

const files = CORPUS.filter((f) => CAPTIONS[f]);
console.log("embedding captions…");
const textVecs = [];
for (const f of files) textVecs.push(await geminiEmbedText("a photo of " + CAPTIONS[f]));

const results = {};
for (const cond of CONDITIONS) {
  let top1 = 0, margins = 0;
  for (let i = 0; i < files.length; i++) {
    const v = (await embed("gemini", files[i], cond.p)).pooled;
    const scores = textVecs.map((t) => cosineSim(v, t));
    const own = scores[i];
    const bestOther = Math.max(...scores.filter((_, j) => j !== i));
    if (own > bestOther) top1++;
    margins += own - bestOther;
  }
  results[cond.name] = { top1Rate: top1 / files.length, meanMargin: margins / files.length };
  console.log(`${cond.name.padEnd(12)} own-caption top-1 ${top1}/${files.length}  mean margin ${fmt(margins / files.length)}`);
}
fs.writeFileSync("results-gemini-alignment.json", JSON.stringify({ model: "gemini-embedding-2", results }, null, 2));
console.log("\nwritten results-gemini-alignment.json");
