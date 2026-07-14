// Why does Florence-2 look rotation-fragile in vector-viz (90° cos 0.56) but robust in the
// VLM playground test (90° cos 0.81)? Suspect: vector-viz renders a 224×224 letterbox first
// (the lab's shared pipeline), but Florence's processor natively resizes to 768×768 — the
// playground fed it full-res. This probe embeds the same image both ways at several angles.
import { Florence2ForConditionalGeneration, AutoProcessor, RawImage } from "@huggingface/transformers";
import { createCanvas, loadImage } from "canvas";
import { poolEmbedding, renderTransform, cosineSim, SIZE } from "../lib/experiment.mjs";

const MODEL_ID = "onnx-community/Florence-2-base-ft";
const IMG = "test-images/photo-forest-path.jpg";
const ANGLES = [0, 45, 90, 180, 270];

const model = await Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
  dtype: { embed_tokens: "fp32", vision_encoder: "fp32", encoder_model: "q8", decoder_model_merged: "q8" },
});
const processor = await AutoProcessor.from_pretrained(MODEL_ID);

async function embedCanvas(c) {
  const { data, width, height } = c.getContext("2d").getImageData(0, 0, c.width, c.height);
  const raw = new RawImage(new Uint8ClampedArray(data), width, height, 4);
  const vin = await processor(raw);
  const feats = await model.encode_image({ pixel_values: vin.pixel_values });
  return poolEmbedding(feats.data, feats.dims);
}

const img = await loadImage(IMG);
console.log(`image ${img.width}×${img.height}, Florence processor native size:`, processor.image_processor?.size ?? "(unknown)");

// Path A — full-res rotate (what the VLM playground effectively did): rotate on a canvas at the
// image's own resolution, no downscale before the processor's native resize.
async function pathA(angle) {
  const side = Math.min(img.width, img.height);
  const c = createCanvas(side, side);
  renderTransform(c.getContext("2d"), img, angle ? { rotation: angle } : {}, side);
  return embedCanvas(c);
}

// Path B — the lab's 224 letterbox first (current vector-viz behavior).
async function pathB(angle) {
  const c = createCanvas(SIZE, SIZE);
  renderTransform(c.getContext("2d"), img, angle ? { rotation: angle } : {}, SIZE);
  return embedCanvas(c);
}

// Path C — render the transform at Florence's native 768 (proposed fix).
async function pathC(angle) {
  const c = createCanvas(768, 768);
  renderTransform(c.getContext("2d"), img, angle ? { rotation: angle } : {}, 768);
  return embedCanvas(c);
}

for (const [name, fn] of [["A full-res", pathA], ["B 224-letterbox", pathB], ["C 768-render", pathC]]) {
  const base = await fn(0);
  const out = [];
  for (const a of ANGLES.slice(1)) out.push(`${a}°=${cosineSim(base, await fn(a)).toFixed(3)}`);
  console.log(`${name.padEnd(16)} ${out.join("  ")}`);
}
