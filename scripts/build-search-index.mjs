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
import { renderTransform, poolEmbedding } from "../lib/experiment.mjs";

const MODELS = {
  clip: { id: "Xenova/clip-vit-base-patch32", size: 224, dim: 512 },
  siglip: { id: "Xenova/siglip-base-patch16-224", size: 224, dim: 768 },
  dinov2: { id: "Xenova/dinov2-small", size: 224, dim: 384 },
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

for (const f of files) {
  db.run(`INSERT OR IGNORE INTO images (file, source) VALUES (?, ?)`, [f, `https://picsum.photos/seed/iel-${parseInt(f.slice(4, 8))}/384`]);
}

const T = await import("@huggingface/transformers");
for (const [m, cfg] of Object.entries(MODELS)) {
  const missing = db.exec(
    `SELECT i.id, i.file FROM images i LEFT JOIN emb_${m} e ON e.image_id = i.id WHERE e.image_id IS NULL ORDER BY i.id`
  )[0];
  if (!missing) { console.log(`${m}: complete`); continue; }
  console.log(`${m}: embedding ${missing.values.length} images`);
  const extractor = await T.pipeline("image-feature-extraction", cfg.id);
  let n = 0;
  for (const [id, file] of missing.values) {
    const img = await loadImage(path.join(DIR, file));
    const c = createCanvas(cfg.size, cfg.size);
    renderTransform(c.getContext("2d"), img, {}, cfg.size);
    const { data, width, height } = c.getContext("2d").getImageData(0, 0, cfg.size, cfg.size);
    const out = await extractor(new T.RawImage(new Uint8ClampedArray(data), width, height, 4));
    const vec = poolEmbedding(out.data, out.dims);           // unit-normalized Float64
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
