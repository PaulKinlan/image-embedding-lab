#!/usr/bin/env node
/**
 * Tiling experiment — does AnyRes tiling make the encoder see more detail?
 *
 * For each model × tiling level (1×1 / 2×2 / 3×3), computes:
 *  - the different-image FLOOR (cosine between unrelated originals), split into "detail" images
 *    (webpages, text, charts, diagrams, maps, signs, logos) and "photo" images. If tiling helps
 *    the model read fine detail, the detail-image floor should DROP as tiling increases (the
 *    documents stop looking identical).
 *  - rotation / crop / flip invariance means, to see how tiling shifts robustness.
 *
 * Writes tiling-results.json. Run: node scripts/tiling-experiment.mjs
 */
import { env, pipeline, RawImage } from "@huggingface/transformers";
import { createCanvas, loadImage } from "canvas";
import { readdirSync } from "node:fs";
import {
  cosineSim,
  makeTiles,
  meanPoolVectors,
  poolEmbedding,
  renderTransform,
  TILE,
} from "../lib/experiment.mjs";

const MODELS = {
  clip: "Xenova/clip-vit-base-patch32",
  siglip: "Xenova/siglip-base-patch16-224",
  dinov2: "Xenova/dinov2-small",
};
const TILINGS = [1, 2, 3];
const KEY = [
  { name: "rot", p: { rotation: 90 } },
  { name: "crop", p: { crop: 0.35 } },
  { name: "flip", p: { flip: "horizontal" } },
];

const DIR = new URL("../test-images/", import.meta.url).pathname;
const files = readdirSync(DIR).filter((f) => /\.(jpg|jpeg|png)$/i.test(f));
const isDetail = (f) => /(webpage|text|chart|diagram|graph|map|sign|logo)/i.test(f);

async function embedImage(extractor, img, params, grid) {
  const renderSize = grid * TILE;
  const canvas = createCanvas(renderSize, renderSize);
  renderTransform(canvas.getContext("2d"), img, params, renderSize);
  const embs = [];
  for (const tile of makeTiles(canvas, grid, createCanvas)) {
    const { data, width, height } = tile.getContext("2d").getImageData(0, 0, tile.width, tile.height);
    const out = await extractor(new RawImage(data, width, height, 4));
    embs.push(poolEmbedding(out.data, out.dims));
  }
  return embs.length === 1 ? embs[0] : meanPoolVectors(embs);
}

function floor(embs) {
  const sims = [];
  for (let i = 0; i < embs.length; i++) {
    for (let j = i + 1; j < embs.length; j++) sims.push(cosineSim(embs[i], embs[j]));
  }
  return sims.length ? sims.reduce((a, b) => a + b, 0) / sims.length : null;
}
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

const results = { images: files.length, detail: files.filter(isDetail).length, models: {} };

for (const [key, id] of Object.entries(MODELS)) {
  process.stderr.write(`\n=== ${key} (${id}) ===\n`);
  env.allowLocalModels = false;
  const extractor = await pipeline("image-feature-extraction", id);
  const imgs = [];
  for (const f of files) imgs.push({ f, img: await loadImage(DIR + f), detail: isDetail(f) });

  const tilings = {};
  for (const grid of TILINGS) {
    process.stderr.write(`  tiling ${grid}×${grid}…\n`);
    const orig = [];
    const inv = { rot: [], crop: [], flip: [] };
    for (const { img, detail } of imgs) {
      const oe = await embedImage(extractor, img, {}, grid);
      orig.push({ oe, detail });
      for (const t of KEY) inv[t.name].push(cosineSim(oe, await embedImage(extractor, img, t.p, grid)));
    }
    tilings[grid] = {
      floorAll: floor(orig.map((o) => o.oe)),
      floorDetail: floor(orig.filter((o) => o.detail).map((o) => o.oe)),
      floorPhoto: floor(orig.filter((o) => !o.detail).map((o) => o.oe)),
      rot: mean(inv.rot),
      crop: mean(inv.crop),
      flip: mean(inv.flip),
    };
    const t = tilings[grid];
    process.stderr.write(
      `    floor all/detail/photo: ${(t.floorAll * 100).toFixed(0)}/${(t.floorDetail * 100).toFixed(0)}/` +
        `${(t.floorPhoto * 100).toFixed(0)}%  rot ${(t.rot * 100).toFixed(0)}%\n`,
    );
  }
  results.models[key] = tilings;
}

console.log(JSON.stringify(results, null, 2));
