#!/usr/bin/env node
/**
 * Image Embedding Lab — CLI
 * 
 * Tests whether image embeddings are invariant to transformations (rotation, crop, scale, flip).
 * Runs CLIP, SigLIP, or DINOv2 via Transformers.js in Node.js.
 * 
 * Usage:
 *   node cli.mjs <image> [--model clip|siglip|dinov2] [--json]
 *   node cli.mjs cat.jpg --model dinov2
 *   node cli.mjs *.jpg --model clip --json > results.json
 * 
 * Install:
 *   npm install @huggingface/transformers canvas
 *   # or: npx image-embedding-lab cat.jpg
 */

import { pipeline, env, RawImage } from '@huggingface/transformers';
import { createCanvas } from 'canvas';
import { loadImage } from 'canvas';
import { readFileSync, writeFileSync } from 'fs';

const MODELS = {
  clip: 'Xenova/clip-vit-base-patch32',
  siglip: 'Xenova/siglip-base-patch16-224',
  dinov2: 'Xenova/dinov2-small',
};

const TRANSFORMS = [
  { name: 'Identity', params: { rotation: 0, crop: 0, scale: 1, flip: 'none' } },
  { name: 'Rotate 90°', params: { rotation: 90, crop: 0, scale: 1, flip: 'none' } },
  { name: 'Rotate 180°', params: { rotation: 180, crop: 0, scale: 1, flip: 'none' } },
  { name: 'Rotate 270°', params: { rotation: 270, crop: 0, scale: 1, flip: 'none' } },
  { name: 'Rotate 15°', params: { rotation: 15, crop: 0, scale: 1, flip: 'none' } },
  { name: 'Rotate 45°', params: { rotation: 45, crop: 0, scale: 1, flip: 'none' } },
  { name: 'Crop 10%', params: { rotation: 0, crop: 0.1, scale: 1, flip: 'none' } },
  { name: 'Crop 20%', params: { rotation: 0, crop: 0.2, scale: 1, flip: 'none' } },
  { name: 'Crop 35%', params: { rotation: 0, crop: 0.35, scale: 1, flip: 'none' } },
  { name: 'Scale 80%', params: { rotation: 0, crop: 0, scale: 0.8, flip: 'none' } },
  { name: 'Scale 120%', params: { rotation: 0, crop: 0, scale: 1.2, flip: 'none' } },
  { name: 'Scale 150%', params: { rotation: 0, crop: 0, scale: 1.5, flip: 'none' } },
  { name: 'Flip H', params: { rotation: 0, crop: 0, scale: 1, flip: 'horizontal' } },
  { name: 'Flip V', params: { rotation: 0, crop: 0, scale: 1, flip: 'vertical' } },
  { name: 'Rotate 90° + Crop 20%', params: { rotation: 90, crop: 0.2, scale: 1, flip: 'none' } },
];

function parseArgs(argv) {
  const args = { files: [], model: 'clip', json: false, verbose: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--model' || a === '-m') args.model = argv[++i];
    else if (a === '--json') args.json = true;
    else if (a === '--verbose' || a === '-v') args.verbose = true;
    else if (a === '--help' || a === '-h') {
      console.log(`Usage: node cli.mjs <image> [options]

Options:
  -m, --model <name>   Model: clip, siglip, or dinov2 (default: clip)
  --json               Output results as JSON
  -v, --verbose        Show progress details
  -h, --help           Show this help

Example:
  node cli.mjs cat.jpg --model dinov2
  node cli.mjs *.jpg --model clip --json > results.json`);
      process.exit(0);
    } else {
      args.files.push(a);
    }
  }
  return args;
}

function cosineSim(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; magA += a[i] * a[i]; magB += b[i] * b[i]; }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function renderTransform(img, params, size = 224) {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, size, size);

  ctx.save();
  ctx.translate(size / 2, size / 2);
  if (params.rotation) ctx.rotate(params.rotation * Math.PI / 180);
  if (params.flip === 'horizontal') ctx.scale(-1, 1);
  if (params.flip === 'vertical') ctx.scale(1, -1);

  const sw = img.width * (1 - params.crop * 2);
  const sh = img.height * (1 - params.crop * 2);
  const sx = img.width * params.crop;
  const sy = img.height * params.crop;
  const aspect = sw / sh;
  let dw = size * params.scale, dh = size * params.scale;
  if (aspect > 1) dh = dw / aspect; else dw = dh * aspect;
  ctx.drawImage(img, sx, sy, sw, sh, -dw / 2, -dh / 2, dw, dh);
  ctx.restore();
  return canvas;
}

async function getEmbedding(extractor, canvas) {
  // RawImage from canvas pixel data (works in Node.js)
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const rawImage = new RawImage(imageData.data, canvas.width, canvas.height, 4); // RGBA
  const result = await extractor(rawImage);
  const flat = result.data || result.tolist()[0];
  return Float64Array.from(flat);
}

function bar(pct, width = 30) {
  const filled = Math.round(pct / 100 * width);
  return '█'.repeat(filled) + '░'.repeat(width - filled);
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.files.length === 0) {
    console.error('Error: no image file specified. Use --help for usage.');
    process.exit(1);
  }

  const modelId = MODELS[args.model];
  if (!modelId) {
    console.error(`Error: unknown model "${args.model}". Choose: clip, siglip, dinov2`);
    process.exit(1);
  }

  // Load model
  const label = args.model.toUpperCase();
  process.stderr.write(`Loading ${label} (${modelId})…\n`);
  env.allowLocalModels = false;
  const extractor = await pipeline('image-feature-extraction', modelId, {
    progress_callback: args.verbose ? (data) => {
      if (data.status === 'progress' && data.total) {
        process.stderr.write(`  ${data.file}: ${Math.round(data.loaded / data.total * 100)}%\r`);
      } else if (data.status === 'done') {
        process.stderr.write(`  ${data.file}: done\n`);
      }
    } : undefined,
  });
  process.stderr.write(`${label} loaded.\n\n`);

  const allResults = [];

  for (const filePath of args.files) {
    process.stderr.write(`Processing: ${filePath}\n`);
    const img = await loadImage(filePath);
    process.stderr.write(`  Image: ${img.width}×${img.height}\n`);

    // Compute original embedding
    const origCanvas = renderTransform(img, { rotation: 0, crop: 0, scale: 1, flip: 'none' });
    const origEmb = await getEmbedding(extractor, origCanvas);

    // Test each transform
    const results = [];
    for (const t of TRANSFORMS) {
      const canvas = renderTransform(img, t.params);
      const emb = await getEmbedding(extractor, canvas);
      const sim = cosineSim(origEmb, emb);
      const pct = Math.round(sim * 100);
      results.push({ name: t.name, similarity: sim, pct });
    }

    allResults.push({ file: filePath, model: args.model, results });

    // Print table
    if (!args.json) {
      console.log(`\n${filePath} (${label})`);
      console.log('─'.repeat(52));
      for (const r of results) {
        const color = r.pct >= 95 ? '\x1b[32m' : r.pct >= 80 ? '\x1b[33m' : '\x1b[31m';
        const reset = '\x1b[0m';
        console.log(`  ${r.name.padEnd(25)} ${color}${bar(r.pct)} ${r.pct}%${reset}`);
      }
      // summary
      const rotAvg = ['Rotate 90°', 'Rotate 180°', 'Rotate 270°'].map(n => results.find(r => r.name === n)?.pct).filter(v => v != null);
      const cropAvg = ['Crop 20%', 'Crop 35%'].map(n => results.find(r => r.name === n)?.pct).filter(v => v != null);
      const flipH = results.find(r => r.name === 'Flip H')?.pct;
      const rotAvgVal = rotAvg.length ? Math.round(rotAvg.reduce((a, b) => a + b, 0) / rotAvg.length) : '—';
      const cropAvgVal = cropAvg.length ? Math.round(cropAvg.reduce((a, b) => a + b, 0) / cropAvg.length) : '—';
      console.log('─'.repeat(52));
      console.log(`  Rotation avg: ${rotAvgVal}%  ·  Crop avg: ${cropAvgVal}%  ·  Flip H: ${flipH != null ? flipH + '%' : '—'}`);
    }
  }

  if (args.json) {
    console.log(JSON.stringify(allResults, null, 2));
  }
}

main().catch(e => { console.error('Error:', e.message); process.exit(1); });
