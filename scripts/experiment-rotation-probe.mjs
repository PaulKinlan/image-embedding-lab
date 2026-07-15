// Experiment 2 — ORIENTATION LINEAR PROBE. Even when cosine stays high under rotation, is the
// rotation angle still RECOVERABLE from the embedding? Train a softmax probe to predict
// {0,90,180,270}° with leave-one-image-out cross-validation. High cosine + decodable angle
// means "insensitive but not information-losing" — invariance and information are different
// claims. Chance = 25%.
import fs from "node:fs";
import { MODELS, CORPUS, embed, fmt } from "./lib-harness.mjs";

const ANGLES = [0, 90, 180, 270];

function trainSoftmax(X, y, classes, { epochs = 400, lr = 0.5, l2 = 1e-3 } = {}) {
  const D = X[0].length, K = classes;
  const W = Array.from({ length: K }, () => new Float64Array(D));
  const b = new Float64Array(K);
  for (let e = 0; e < epochs; e++) {
    const gW = Array.from({ length: K }, () => new Float64Array(D));
    const gb = new Float64Array(K);
    for (let n = 0; n < X.length; n++) {
      const logits = new Float64Array(K);
      for (let k = 0; k < K; k++) {
        let s = b[k];
        for (let d = 0; d < D; d++) s += W[k][d] * X[n][d];
        logits[k] = s;
      }
      const mx = Math.max(...logits);
      let Z = 0;
      const p = new Float64Array(K);
      for (let k = 0; k < K; k++) { p[k] = Math.exp(logits[k] - mx); Z += p[k]; }
      for (let k = 0; k < K; k++) {
        const err = p[k] / Z - (y[n] === k ? 1 : 0);
        for (let d = 0; d < D; d++) gW[k][d] += err * X[n][d];
        gb[k] += err;
      }
    }
    for (let k = 0; k < K; k++) {
      for (let d = 0; d < D; d++) W[k][d] -= lr * (gW[k][d] / X.length + l2 * W[k][d]);
      b[k] -= lr * gb[k] / X.length;
    }
  }
  return (x) => {
    let best = 0, bestS = -Infinity;
    for (let k = 0; k < classes; k++) {
      let s = b[k];
      for (let d = 0; d < x.length; d++) s += W[k][d] * x[d];
      if (s > bestS) { bestS = s; best = k; }
    }
    return best;
  };
}

const results = {};
for (const model of Object.keys(MODELS)) {
  console.log(`\n=== ${model} ===`);
  // Gather all embeddings first (cached)
  const data = [];   // { file, angleIdx, vec }
  for (const file of CORPUS) {
    for (let a = 0; a < ANGLES.length; a++) {
      const p = ANGLES[a] ? { rotation: ANGLES[a] } : {};
      data.push({ file, a, vec: (await embed(model, file, p)).pooled });
    }
  }
  // Leave-one-image-out CV
  let correct = 0, total = 0;
  for (const heldOut of CORPUS) {
    const train = data.filter((d) => d.file !== heldOut);
    const test = data.filter((d) => d.file === heldOut);
    const predict = trainSoftmax(train.map((d) => d.vec), train.map((d) => d.a), ANGLES.length);
    for (const t of test) { if (predict(t.vec) === t.a) correct++; total++; }
  }
  const acc = correct / total;
  results[model] = { accuracy: acc, chance: 0.25, n: total };
  console.log(`probe accuracy ${fmt(acc)} (chance 0.250, n=${total})`);
}
fs.writeFileSync("results-rotation-probe.json", JSON.stringify({ angles: ANGLES, corpus: CORPUS.length, results }, null, 2));
console.log("\nwritten results-rotation-probe.json");
