// Experiment 4 — SAME CONTENT, SWAPPED LAYOUT. Composite two images side by side as A|B and
// B|A: identical content, different arrangement. cos(A|B, B|A) close to 1 → layout carries
// little weight; close to the composite floor → layout dominates. Floor here = mean cosine
// between composites made of DIFFERENT image pairs.
import fs from "node:fs";
import { createCanvas } from "canvas";
import { MODELS, getImage, cosineSim, meanStd, fmt } from "./lib-harness.mjs";
import { poolEmbedding } from "../lib/experiment.mjs";

const PAIRS = [
  ["photo-forest-path.jpg", "photo-ocean-waves.jpg"],
  ["photo-cafe.jpg", "photo-mountain-lake.jpg"],
  ["photo-skateboard.jpg", "photo-floating-market.jpg"],
  ["webpage-hn.png", "webpage-wikipedia.png"],
  ["text-readme.png", "webpage-example.png"],
  ["photo-sea-cliff.jpg", "webpage-sotw.png"],
];

// Cover-fit half-canvas composite: left = a, right = b.
function composite(imgA, imgB, size) {
  const c = createCanvas(size, size);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#7f7f7f";
  ctx.fillRect(0, 0, size, size);
  const half = size / 2;
  for (const [img, x0] of [[imgA, 0], [imgB, half]]) {
    const s = Math.min(img.width / half, img.height / size);
    const sw = half * s, sh = size * s;
    ctx.drawImage(img, (img.width - sw) / 2, (img.height - sh) / 2, sw, sh, x0, 0, half, size);
  }
  return c;
}

const T = await import("@huggingface/transformers");
async function makeEmbedder(model) {
  const m = MODELS[model];
  if (m.api) {
    const { geminiEmbedCanvas } = await import("./lib-harness.mjs");
    return geminiEmbedCanvas;
  }
  if (model === "florence") {
    const fl = await T.Florence2ForConditionalGeneration.from_pretrained(m.id, {
      dtype: { embed_tokens: "fp32", vision_encoder: "fp32", encoder_model: "q8", decoder_model_merged: "q8" },
    });
    const processor = await T.AutoProcessor.from_pretrained(m.id);
    return async (c) => {
      const { data, width, height } = c.getContext("2d").getImageData(0, 0, c.width, c.height);
      const vin = await processor(new T.RawImage(new Uint8ClampedArray(data), width, height, 4));
      const feats = await fl.encode_image({ pixel_values: vin.pixel_values });
      return poolEmbedding(feats.data, feats.dims);
    };
  }
  const extractor = await T.pipeline("image-feature-extraction", m.id);
  return async (c) => {
    const { data, width, height } = c.getContext("2d").getImageData(0, 0, c.width, c.height);
    const out = await extractor(new T.RawImage(new Uint8ClampedArray(data), width, height, 4));
    return poolEmbedding(out.data, out.dims);
  };
}

const results = {};
for (const model of Object.keys(MODELS)) {
  console.log(`\n=== ${model} ===`);
  const embedC = await makeEmbedder(model);
  const size = MODELS[model].size;
  const abVecs = [], swapSims = [];
  for (const [fa, fb] of PAIRS) {
    const a = await getImage(fa), b = await getImage(fb);
    const vAB = await embedC(composite(a, b, size));
    const vBA = await embedC(composite(b, a, size));
    abVecs.push(vAB);
    swapSims.push(cosineSim(vAB, vBA));
  }
  const flSims = [];
  for (let i = 0; i < abVecs.length; i++) {
    for (let j = i + 1; j < abVecs.length; j++) flSims.push(cosineSim(abVecs[i], abVecs[j]));
  }
  const swap = meanStd(swapSims), fl = meanStd(flSims);
  const normalized = (swap.mean - fl.mean) / (1 - fl.mean);
  results[model] = { swap: swap.mean, swapStd: swap.std, floor: fl.mean, normalized };
  console.log(`cos(A|B, B|A) ${fmt(swap.mean)} ±${fmt(swap.std)}  composite floor ${fmt(fl.mean)}  floor-normalized ${fmt(normalized)}`);
}
fs.writeFileSync("results-layout-swap.json", JSON.stringify({ pairs: PAIRS, results }, null, 2));
console.log("\nwritten results-layout-swap.json");
