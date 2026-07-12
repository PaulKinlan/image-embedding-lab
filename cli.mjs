#!/usr/bin/env node
/**
 * Image Embedding Lab — CLI
 *
 * Measures how invariant a vision encoder's embedding is to image transforms (rotation, crop,
 * scale, flip, blur, brightness, occlusion…). Runs CLIP, SigLIP, or DINOv2 via Transformers.js.
 *
 * Every transform is embedded at raw pixels AND at three JPEG quality levels (q0.9 / q0.5 /
 * q0.2), so you can see whether transform-invariance survives increasingly heavy compression.
 * Embeddings are mean-pooled to a single vector (see lib/experiment.mjs poolEmbedding — without
 * this, SigLIP/DINOv2 are compared as flattened patch grids and rotation trivially scrambles
 * them). A random-different-image baseline "floor" is reported so the numbers are interpretable.
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
  VARIANTS,
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

/** Embed a canvas → one pooled, L2-normalized vector. quality=null → raw pixels; a number →
 *  a JPEG round-trip at that quality first (injects real compression noise). */
async function embedVariant(extractor, canvas, quality) {
  let src = canvas;
  if (quality != null) {
    const buf = canvas.toBuffer("image/jpeg", { quality });
    const img = await loadImage(buf);
    src = createCanvas(SIZE, SIZE);
    src.getContext("2d").drawImage(img, 0, 0, SIZE, SIZE);
  }
  const { data, width, height } = src.getContext("2d").getImageData(0, 0, src.width, src.height);
  const out = await extractor(new RawImage(data, width, height, 4));
  return poolEmbedding(out.data, out.dims);
}

function bar(pct, width = 18) {
  const f = Math.max(0, Math.min(width, Math.round((pct / 100) * width)));
  return "█".repeat(f) + "░".repeat(width - f);
}
function color(pct) {
  return pct >= 90 ? "\x1b[32m" : pct >= 70 ? "\x1b[33m" : "\x1b[31m";
}
const pctOf = (x) => (x == null ? "—" : String(Math.round(x * 100)));

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
  const origEmbs = {};
  for (const v of VARIANTS) origEmbs[v.key] = [];

  for (const file of files) {
    process.stderr.write(`Processing ${file}…\n`);
    const img = await loadImage(file);
    const canvas = createCanvas(SIZE, SIZE);
    const ctx = canvas.getContext("2d");

    renderTransform(ctx, img, {}); // Identity
    const orig = {};
    for (const v of VARIANTS) {
      orig[v.key] = await embedVariant(extractor, canvas, v.quality);
      origEmbs[v.key].push(orig[v.key]);
    }

    const results = [];
    for (const t of TRANSFORMS) {
      renderTransform(ctx, img, t.p);
      const row = { name: t.name };
      for (const v of VARIANTS) {
        row[v.key] = cosineSim(orig[v.key], await embedVariant(extractor, canvas, v.quality));
      }
      results.push(row);
    }
    perImage.push({ file, results });
  }

  // Random-different-image baseline floor, per variant.
  const pairFloor = (embs) => {
    const sims = [];
    for (let i = 0; i < embs.length; i++) {
      for (let j = i + 1; j < embs.length; j++) sims.push(cosineSim(embs[i], embs[j]));
    }
    const s = meanStd(sims);
    sims.sort((a, b) => a - b);
    return { ...s, p95: sims.length ? sims[Math.floor(sims.length * 0.95)] : null };
  };
  const baseline = {};
  for (const v of VARIANTS) baseline[v.key] = pairFloor(origEmbs[v.key]);

  // Per-transform summary (mean ± std across images), per variant.
  const summary = TRANSFORMS.map((t) => {
    const row = { name: t.name };
    for (const v of VARIANTS) {
      row[v.key] = meanStd(perImage.map((im) => im.results.find((r) => r.name === t.name)?.[v.key]));
    }
    return row;
  });

  const report = {
    model: args.model,
    modelId,
    images: files.length,
    variants: VARIANTS.map((v) => v.key),
    baseline,
    summary,
    perImage,
  };

  if (args.json) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  // Human table: raw + each JPEG quality.
  console.log(`\n${label} — ${files.length} image(s), mean similarity across images`);
  console.log(`(pooled + L2-normalized embeddings; raw pixels vs JPEG q0.9 / q0.5 / q0.2)\n`);
  const head = "raw".padStart(4) + VARIANTS.slice(1).map((v) => v.label.padStart(9)).join("");
  console.log(`  ${"transform".padEnd(22)} ${head}`);
  console.log("  " + "─".repeat(58));
  for (const s of summary) {
    const rp = Math.round((s.raw.mean ?? 0) * 100);
    const rest = VARIANTS.slice(1)
      .map((v) => `${pctOf(s[v.key].mean).padStart(6)}%`.padStart(9))
      .join("");
    const reset = "\x1b[0m";
    console.log(
      `  ${s.name.padEnd(22)} ${color(rp)}${String(rp).padStart(3)}%${reset} ${bar(rp)}${rest}`,
    );
  }
  console.log("  " + "─".repeat(58));
  const bl = (v) => (baseline[v].mean != null ? `${Math.round(baseline[v].mean * 100)}%` : "—");
  console.log(
    `  Different-image floor — ` + VARIANTS.map((v) => `${v.label} ${bl(v.key)}`).join(" · "),
  );
  console.log(`  (Similarities near the floor mean "as different as an unrelated image".)`);
}

main().catch((e) => {
  console.error("Error:", e.stack || e.message);
  process.exit(1);
});
