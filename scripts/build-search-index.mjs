// Build the reverse-image-search index: embed every image in search-images/ with the three
// browser-runnable encoders and pack everything into ONE SQLite file (search-index.sqlite)
// that search.html loads with sql.js. Embeddings are Float32 BLOBs, unit-normalized, so
// cosine = dot product at query time.
//
// Run: node scripts/build-search-index.mjs        (~40 min cold; resumable — skips rows that
// already exist in an existing search-index.sqlite)
import fs from "node:fs";
import path from "node:path";
import initSqlJs from "sql.js";
import { createCanvas, loadImage } from "canvas";
import { poolEmbedding, fitToSquare } from "../lib/experiment.mjs";

const MODELS = {
  clip: { id: "Xenova/clip-vit-base-patch32", size: 224, dim: 512 },
  siglip: { id: "Xenova/siglip-base-patch16-224", size: 224, dim: 768 },
  dinov2: { id: "Xenova/dinov2-small", size: 224, dim: 384 },
  // Florence-2's DaViT (VLM encoder): vision_encoder fp32 to match the browser's WASM dtype
  // config (vlm.html uses fp32 vision on wasm); the page's calibration check verifies the
  // match end-to-end on the user's machine.
  florence: { id: "onnx-community/Florence-2-base-ft", size: 768, dim: 768, florence: true },
  // NOTE: the Gemma 4 column is built by scripts/build-search-index-gemma4.mjs — it needs
  // onnxruntime-node >= 1.27 (GatherBlockQuantized bits attr), which cannot share a process
  // with transformers.js's nested onnxruntime-node 1.21 (the wrong native .so gets dlopened).
  // Gemini Embedding 2 (hosted API): built with GEMINI_API_KEY; the page asks the user for
  // their own key at query time.
  gemini: { id: "gemini-embedding-2", size: 768, dim: 3072, gemini: true },
};

const DIR = "search-images";
const OUT = "search-index.sqlite";
const files = fs.readdirSync(DIR).filter((f) => f.endsWith(".jpg")).sort();
console.log(`${files.length} corpus images`);

const SQL = await initSqlJs();
const db = fs.existsSync(OUT) ? new SQL.Database(fs.readFileSync(OUT)) : new SQL.Database();
db.run(`CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, file TEXT UNIQUE, source TEXT);`);
for (const m of Object.keys(MODELS)) {
  db.run(`CREATE TABLE IF NOT EXISTS emb_${m} (image_id INTEGER PRIMARY KEY REFERENCES images(id), v BLOB);`);
}
db.run(`CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);`);
db.run(`INSERT OR REPLACE INTO meta VALUES ('models', ?)`, [JSON.stringify(MODELS)]);
db.run(`INSERT OR REPLACE INTO meta VALUES ('license', 'Images via picsum.photos (Unsplash-sourced, Unsplash license)')`);

// ids.txt maps "ordinal picsum-id" (the corpus is fetched by unique catalog id — seeds collide)
const idMap = Object.fromEntries(
  fs.readFileSync(path.join(DIR, "ids.txt"), "utf8").trim().split("\n").map((l) => l.split(" "))
);
for (const f of files) {
  const ord = String(parseInt(f.slice(4, 8)));
  db.run(`INSERT OR IGNORE INTO images (file, source) VALUES (?, ?)`, [f, `https://picsum.photos/id/${idMap[ord]}/384`]);
}

const T = await import("@huggingface/transformers");
for (const [m, cfg] of Object.entries(MODELS)) {
  const missing = db.exec(
    `SELECT i.id, i.file FROM images i LEFT JOIN emb_${m} e ON e.image_id = i.id WHERE e.image_id IS NULL ORDER BY i.id`
  )[0];
  if (!missing) { console.log(`${m}: complete`); continue; }
  console.log(`${m}: embedding ${missing.values.length} images`);
  // dtype q8 for the pipeline models: the browser's WASM backend loads the quantized ONNX by
  // default, and Node's default is fp32 — same pixels through different weights costs ~0.14
  // cosine. The index must use the SAME precision the querying page will.
  let embedRaw;
  if (cfg.gemini) {
    if (!process.env.GEMINI_API_KEY) { console.log("gemini: GEMINI_API_KEY not set, skipping"); continue; }
    embedRaw = { native: async (rgba, canvas) => {
      const b64 = canvas.toBuffer("image/png").toString("base64");
      const res = await fetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent", {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": process.env.GEMINI_API_KEY },
        body: JSON.stringify({ content: { parts: [{ inlineData: { mimeType: "image/png", data: b64 } }] } }),
      });
      if (!res.ok) throw new Error(`gemini ${res.status}: ${(await res.text()).slice(0, 200)}`);
      const values = (await res.json()).embedding.values;
      let mag = 0;
      for (const x of values) mag += x * x;
      mag = Math.sqrt(mag) || 1;
      await new Promise((r) => setTimeout(r, 350));
      return values.map((x) => x / mag);
    } };
  } else if (cfg.florence) {
    const fl = await T.Florence2ForConditionalGeneration.from_pretrained(cfg.id, {
      dtype: { embed_tokens: "fp32", vision_encoder: "fp32", encoder_model: "q8", decoder_model_merged: "q8" },
    });
    const processor = await T.AutoProcessor.from_pretrained(cfg.id);
    embedRaw = async (raw) => {
      const vin = await processor(raw);
      const feats = await fl.encode_image({ pixel_values: vin.pixel_values });
      return poolEmbedding(feats.data, feats.dims);
    };
  } else {
    const extractor = await T.pipeline("image-feature-extraction", cfg.id, { dtype: "q8" });
    embedRaw = async (raw) => {
      const out = await extractor(raw);
      return poolEmbedding(out.data, out.dims);
    };
  }
  let n = 0;
  for (const [id, file] of missing.values) {
    // Draw 1:1 at native size (no runtime resampling), then downscale with the SHARED pure-JS
    // resampler — the browser query path runs the identical code, so identity queries match.
    const img = await loadImage(path.join(DIR, file));
    const c = createCanvas(img.width, img.height);
    c.getContext("2d").drawImage(img, 0, 0);
    const raw = c.getContext("2d").getImageData(0, 0, img.width, img.height);
    let vec;
    if (cfg.gemini) {
      const sq = fitToSquare(raw, cfg.size);                  // shared square render → PNG → API
      const sc = createCanvas(cfg.size, cfg.size);
      const sctx = sc.getContext("2d");
      const idata = sctx.createImageData(cfg.size, cfg.size);
      idata.data.set(sq.data);
      sctx.putImageData(idata, 0, 0);
      vec = await embedRaw.native(sq, sc);
    } else {
      const sq = fitToSquare(raw, cfg.size);
      vec = await embedRaw(new T.RawImage(new Uint8ClampedArray(sq.data), sq.width, sq.height, 4));
    }
    db.run(`INSERT INTO emb_${m} (image_id, v) VALUES (?, ?)`, [id, new Uint8Array(Float32Array.from(vec).buffer)]);
    if (++n % 50 === 0) {
      console.log(`  ${m} ${n}/${missing.values.length}`);
      fs.writeFileSync(OUT, Buffer.from(db.export()));       // checkpoint
    }
  }
  fs.writeFileSync(OUT, Buffer.from(db.export()));
  console.log(`${m}: done`);
}
fs.writeFileSync(OUT, Buffer.from(db.export()));
const stat = fs.statSync(OUT);
console.log(`written ${OUT} (${(stat.size / 1048576).toFixed(1)} MB)`);
