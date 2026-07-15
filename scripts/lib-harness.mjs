// Shared Node harness for the layout-thesis experiment suite (2026-07-15).
//
// Thesis under test: an embedding encodes WHAT-IS-WHERE, not just WHAT — VLM encoders
// (Florence-2/DaViT) preserve spatial layout for their decoder, contrastive encoders
// (CLIP/SigLIP) are trained to discard viewpoint, self-supervised (DINOv2) sits between.
//
// Provides: corpus, model loading (pooled + unpooled + Florence), transform rendering at each
// model's NATIVE resolution (AGENTS.md rule 2), a disk cache of embeddings keyed by
// model×image×transform so the seven experiments never re-embed the same condition, floors,
// and small stats helpers.
import fs from "node:fs";
import path from "node:path";
import { createCanvas, loadImage } from "canvas";
import { renderTransform, poolEmbedding, cosineSim, meanStd, SIZE } from "../lib/experiment.mjs";

export { cosineSim, meanStd };

export const MODELS = {
  clip: { id: "Xenova/clip-vit-base-patch32", size: 224 },
  siglip: { id: "Xenova/siglip-base-patch16-224", size: 224 },
  dinov2: { id: "Xenova/dinov2-small", size: 224 },
  florence: { id: "onnx-community/Florence-2-base-ft", size: 768 },
};

export const CORPUS = JSON.parse(fs.readFileSync(new URL("../test-images/manifest.json", import.meta.url)))
  .filter((f) => f !== "manifest.json");

export const PHOTO_FILES = CORPUS.filter((f) => f.startsWith("photo-"));
export const TEXTUAL_FILES = CORPUS.filter((f) => f.startsWith("text-") || f.startsWith("webpage-"));

const CACHE_DIR = new URL("../.experiment-cache/", import.meta.url).pathname;
fs.mkdirSync(CACHE_DIR, { recursive: true });

const imageCache = new Map();
export async function getImage(file) {
  if (!imageCache.has(file)) {
    imageCache.set(file, await loadImage(path.join(path.dirname(CACHE_DIR), "test-images", file)));
  }
  return imageCache.get(file);
}

// --- Model runtimes (lazy) ---
const runtimes = {};

async function getRuntime(model) {
  if (runtimes[model]) return runtimes[model];
  const T = await import("@huggingface/transformers");
  const m = MODELS[model];
  if (model === "florence") {
    const fl = await T.Florence2ForConditionalGeneration.from_pretrained(m.id, {
      dtype: { embed_tokens: "fp32", vision_encoder: "fp32", encoder_model: "q8", decoder_model_merged: "q8" },
    });
    const processor = await T.AutoProcessor.from_pretrained(m.id);
    const tokenizer = await T.AutoTokenizer.from_pretrained(m.id);
    runtimes[model] = {
      async embedCanvas(c) {
        const raw = canvasToRaw(c, T);
        const vin = await processor(raw);
        const feats = await fl.encode_image({ pixel_values: vin.pixel_values });
        return { pooled: poolEmbedding(feats.data, feats.dims), tokens: Array.from(feats.data), dims: feats.dims };
      },
      async caption(c, task = "<CAPTION>") {
        const raw = canvasToRaw(c, T);
        const prompts = processor.construct_prompts(task);
        const visionInputs = await processor(raw);
        const textInputs = tokenizer(prompts);
        const ids = await fl.generate({ ...textInputs, ...visionInputs, max_new_tokens: 64, num_beams: 1, do_sample: false });
        const text = tokenizer.batch_decode(ids, { skip_special_tokens: false })[0];
        return (processor.post_process_generation(text, task, raw.size)[task] || "").trim();
      },
    };
  } else {
    const extractor = await T.pipeline("image-feature-extraction", m.id);
    runtimes[model] = {
      async embedCanvas(c) {
        const raw = canvasToRaw(c, T);
        const out = await extractor(raw);
        return { pooled: poolEmbedding(out.data, out.dims), tokens: Array.from(out.data), dims: out.dims };
      },
    };
  }
  return runtimes[model];
}

function canvasToRaw(c, T) {
  const { data, width, height } = c.getContext("2d").getImageData(0, 0, c.width, c.height);
  return new T.RawImage(new Uint8ClampedArray(data), width, height, 4);
}

export function renderFile(img, model, p = {}) {
  const size = MODELS[model].size;
  const c = createCanvas(size, size);
  renderTransform(c.getContext("2d"), img, p, size);
  return c;
}

// --- Cached embedding: (model, file, transform) → { pooled, dims [, tokens] } ---
// Token grids are big; cache them only when withTokens is requested.
function cachePath(model) { return path.join(CACHE_DIR, `${model}.json`); }
const diskCache = {};
function loadCache(model) {
  if (!diskCache[model]) {
    try { diskCache[model] = JSON.parse(fs.readFileSync(cachePath(model), "utf8")); }
    catch { diskCache[model] = {}; }
  }
  return diskCache[model];
}
function saveCache(model) {
  fs.writeFileSync(cachePath(model), JSON.stringify(diskCache[model]));
}

const tokenMemCache = new Map();   // token grids are huge — memory-only, never written to disk

export async function embed(model, file, p = {}, { withTokens = false } = {}) {
  const key = `${file}|${JSON.stringify(p)}`;
  if (withTokens) {
    const memKey = `${model}|${key}`;
    if (tokenMemCache.has(memKey)) return tokenMemCache.get(memKey);
    const rt = await getRuntime(model);
    const img = await getImage(file);
    const res = await rt.embedCanvas(renderFile(img, model, p));
    const entry = { pooled: Array.from(res.pooled), tokens: res.tokens, dims: res.dims };
    tokenMemCache.set(memKey, entry);
    return entry;
  }
  const cache = loadCache(model);
  if (cache[key]) return cache[key];
  const rt = await getRuntime(model);
  const img = await getImage(file);
  const res = await rt.embedCanvas(renderFile(img, model, p));
  const entry = { pooled: Array.from(res.pooled), dims: res.dims };
  cache[key] = entry;
  saveCache(model);
  return entry;
}

export async function caption(file, p = {}) {
  const key = `caption|${file}|${JSON.stringify(p)}`;
  const cache = loadCache("florence");
  if (cache[key]) return cache[key];
  const rt = await getRuntime("florence");
  const img = await getImage(file);
  const text = await rt.caption(renderFile(img, "florence", p));
  cache[key] = text;
  saveCache("florence");
  return text;
}

/** Different-image floor: mean pairwise cosine between identity embeddings of distinct files. */
export async function floor(model, files = CORPUS) {
  const vecs = [];
  for (const f of files) vecs.push((await embed(model, f)).pooled);
  const sims = [];
  for (let i = 0; i < vecs.length; i++) {
    for (let j = i + 1; j < vecs.length; j++) sims.push(cosineSim(vecs[i], vecs[j]));
  }
  return meanStd(sims);
}

export function fmt(x, digits = 3) { return x == null ? "  n/a" : x.toFixed(digits); }

/** Floor-normalized robustness: how much of the identity→floor range is retained. */
export function normalizedSim(sim, fl) { return (sim - fl) / (1 - fl); }
