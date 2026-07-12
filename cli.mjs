#!/usr/bin/env node
/**
 * Image Embedding Lab — CLI
 *
 * Measures how invariant a vision encoder's embedding is to image transforms (rotation, crop,
 * scale, flip, blur, brightness, occlusion…). Runs CLIP, SigLIP, or DINOv2 via Transformers.js.
 *
 * Every transform is embedded TWICE — from raw pixels and after a JPEG round-trip (q0.9) — so
 * you can see how much apparent (in)variance is really JPEG compression noise. Embeddings are
 * mean-pooled to a single vector (see lib/experiment.mjs poolEmbedding — without this, SigLIP/
 * DINOv2 are compared as flattened patch grids and rotation trivially scrambles them). A
 * random-different-image baseline "floor" is reported so the similarities are interpretable.
 *
 * Usage:
 *   node cli.mjs <images...> [--model clip|siglip|dinov2] [--json] [--limit N] [--verbose]
 *   node cli.mjs test-images/*.jpg --model dinov2 --json > results-dinov2.json
 */

import { env, pipeline, RawImage } from "@huggingface/transformers";
import { createCanvas, loadImage } from "canvas";
import {
  cosineSim,
  meanStd,
  poolEmbedding,
  renderTransform,
  SIZE,
  TRANSFORMS,
} from "./lib/experiment.mjs";

const MODELS = {
  clip: "Xenova/clip-vit-base-patch32",
  siglip: "Xenova/siglip-base-patch16-224",
  dinov2: "Xenova/dinov2-small",
};

function parseArgs(argv) {
  const args = { files: [], model: "clip", json: false, verbose: false, limit: Infinity };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--model" || a === "-m") args.model = argv[++i];
    else if (a === "--json") args.json = true;
    else if (a === "--limit") args.limit = Number(argv[++i]);
    else if (a === "--verbose" || a === "-v") args.verbose = true;
    else if (a === "--help" || a === "-h") {
      console.log(`Usage: node cli.mjs <images...> [options]

Options:
  -m, --model <name>   clip | siglip | dinov2 (default: clip)
  --limit <n>          Only process the first n images (quick runs)
  --json               Output results as JSON
  -v, --verbose        Model download progress
  -h, --help           Show this help`);
      process.exit(0);
    } else args.files.push(a);
  }
  return args;
}

/** Embed a canvas's raw pixels: RawImage → model → single pooled, normalized vector. */
async function embedRaw(extractor, canvas) {
  const ctx = canvas.getContext("2d");
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const raw = new RawImage(data, width, height, 4);
  const out = await extractor(raw);
  return poolEmbedding(out.data, out.dims);
}

/** Embed a canvas after a JPEG round-trip (q0.9) — injects real compression noise. */
async function embedJpeg(extractor, canvas) {
  const buf = canvas.toBuffer("image/jpeg", { quality: 0.9 });
  const img = await loadImage(buf);
  const tmp = createCanvas(SIZE, SIZE);
  tmp.getContext("2d").drawImage(img, 0, 0, SIZE, SIZE);
  return embedRaw(extractor, tmp);
}

function bar(pct, width = 22) {
  const f = Math.max(0, Math.min(width, Math.round((pct / 100) * width)));
  return "█".repeat(f) + "░".repeat(width - f);
}
function color(pct) {
  return pct >= 90 ? "\x1b[32m" : pct >= 70 ? "\x1b[33m" : "\x1b[31m";
}

async function main() {
  const args = parseArgs(process.argv);
  if (!args.files.length) {
    console.error("Error: no image files. Use --help.");
    process.exit(1);
  }
  const modelId = MODELS[args.model];
  if (!modelId) {
    console.error(`Error: unknown model "${args.model}". Choose: clip, siglip, dinov2.`);
    process.exit(1);
  }
  const files = args.files.slice(0, args.limit);
  const label = args.model.toUpperCase();

  env.allowLocalModels = false;
  process.stderr.write(`Loading ${label} (${modelId})…\n`);
  const extractor = await pipeline("image-feature-extraction", modelId, {
    progress_callback: args.verbose
      ? (d) => {
        if (d.status === "progress" && d.total) {
          process.stderr.write(`  ${d.file}: ${Math.round((d.loaded / d.total) * 100)}%\r`);
        } else if (d.status === "done") process.stderr.write(`  ${d.file}: done\n`);
      }
      : undefined,
  });
  process.stderr.write(`${label} loaded.\n\n`);

  const perImage = [];
  const origRawEmbs = [];
  const origJpegEmbs = [];

  for (const file of files) {
    process.stderr.write(`Processing ${file}…\n`);
    const img = await loadImage(file);
    const canvas = createCanvas(SIZE, SIZE);
    const ctx = canvas.getContext("2d");

    renderTransform(ctx, img, {}); // Identity
    const origRaw = await embedRaw(extractor, canvas);
    const origJpeg = await embedJpeg(extractor, canvas);
    origRawEmbs.push(origRaw);
    origJpegEmbs.push(origJpeg);

    const results = [];
    for (const t of TRANSFORMS) {
      renderTransform(ctx, img, t.p);
      const raw = cosineSim(origRaw, await embedRaw(extractor, canvas));
      const jpeg = cosineSim(origJpeg, await embedJpeg(extractor, canvas));
      results.push({ name: t.name, raw, jpeg });
    }
    perImage.push({ file, results });
  }

  // Random-different-image baseline: cosine between every pair of distinct originals.
  const pairFloor = (embs) => {
    const sims = [];
    for (let i = 0; i < embs.length; i++) {
      for (let j = i + 1; j < embs.length; j++) sims.push(cosineSim(embs[i], embs[j]));
    }
    const s = meanStd(sims);
    sims.sort((a, b) => a - b);
    return { ...s, p95: sims.length ? sims[Math.floor(sims.length * 0.95)] : null };
  };
  const baseline = { raw: pairFloor(origRawEmbs), jpeg: pairFloor(origJpegEmbs) };

  // Per-transform summary (mean ± std across images), raw and jpeg.
  const summary = TRANSFORMS.map((t) => {
    const raw = meanStd(perImage.map((im) => im.results.find((r) => r.name === t.name)?.raw));
    const jpeg = meanStd(perImage.map((im) => im.results.find((r) => r.name === t.name)?.jpeg));
    return { name: t.name, raw, jpeg };
  });

  const report = { model: args.model, modelId, images: files.length, baseline, summary, perImage };

  if (args.json) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  // Human table.
  console.log(`\n${label} — ${files.length} image(s), mean similarity across images`);
  console.log(`(pooled + L2-normalized embeddings; each transform embedded raw and via JPEG)\n`);
  console.log(`  ${"transform".padEnd(22)} ${"raw".padStart(5)}  ${"jpeg".padStart(5)}`);
  console.log("  " + "─".repeat(52));
  for (const s of summary) {
    const rp = Math.round((s.raw.mean ?? 0) * 100);
    const jp = Math.round((s.jpeg.mean ?? 0) * 100);
    const reset = "\x1b[0m";
    console.log(
      `  ${s.name.padEnd(22)} ${color(rp)}${String(rp).padStart(3)}%${reset} ${bar(rp)}  ` +
        `${color(jp)}${String(jp).padStart(3)}%${reset}`,
    );
  }
  console.log("  " + "─".repeat(52));
  const bl = (b) => (b.mean != null ? `${Math.round(b.mean * 100)}% (p95 ${Math.round(b.p95 * 100)}%)` : "—");
  console.log(`  Different-image floor — raw: ${bl(baseline.raw)} · jpeg: ${bl(baseline.jpeg)}`);
  console.log(`  (Similarities near the floor mean "as different as an unrelated image".)`);
}

main().catch((e) => {
  console.error("Error:", e.stack || e.message);
  process.exit(1);
});
