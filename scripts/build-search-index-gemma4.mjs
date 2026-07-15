// Standalone Gemma 4 column builder. Separate from build-search-index.mjs because the q4
// vision encoder needs onnxruntime-node >= 1.27 (GatherBlockQuantized with a `bits`
// attribute), and 1.27's native binding cannot share a process with transformers.js's nested
// onnxruntime-node 1.21 — the loader dlopens the wrong libonnxruntime.so.
//
// Downloads the q4 model into GEMMA4_DIR if missing. Resumable like the main builder.
import fs from "node:fs";
import path from "node:path";
import initSqlJs from "sql.js";
import ort from "onnxruntime-node";
import { createCanvas, loadImage } from "canvas";
import { gemma4Embed } from "../lib/gemma4.mjs";

const DIR = "search-images";
const OUT = "search-index.sqlite";
const GEMMA4_DIR = process.env.GEMMA4_DIR || "/tmp/claude-1000/-home-paulkinlan-journal/60f7e06a-d435-4863-a9aa-2722ec61091e/scratchpad/gemma4";
const BASE = "https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx";

fs.mkdirSync(GEMMA4_DIR, { recursive: true });
for (const f of ["vision_encoder_q4.onnx", "vision_encoder_q4.onnx_data"]) {
  const p = path.join(GEMMA4_DIR, f);
  if (!fs.existsSync(p)) {
    console.log("downloading", f);
    const res = await fetch(`${BASE}/${f}`);
    fs.writeFileSync(p, Buffer.from(await res.arrayBuffer()));
  }
}

const SQL = await initSqlJs();
const db = new SQL.Database(fs.readFileSync(OUT));
db.run(`CREATE TABLE IF NOT EXISTS emb_gemma4 (image_id INTEGER PRIMARY KEY REFERENCES images(id), v BLOB);`);
const missing = db.exec(
  `SELECT i.id, i.file FROM images i LEFT JOIN emb_gemma4 e ON e.image_id = i.id WHERE e.image_id IS NULL ORDER BY i.id`
)[0];
if (!missing) { console.log("gemma4: complete"); process.exit(0); }
console.log(`gemma4: embedding ${missing.values.length} images`);

const session = await ort.InferenceSession.create(path.join(GEMMA4_DIR, "vision_encoder_q4.onnx"), { executionProviders: ["cpu"] });
let n = 0;
for (const [id, file] of missing.values) {
  const img = await loadImage(path.join(DIR, file));
  const c = createCanvas(img.width, img.height);
  c.getContext("2d").drawImage(img, 0, 0);
  const raw = c.getContext("2d").getImageData(0, 0, img.width, img.height);
  const vec = await gemma4Embed(session, ort, raw);   // resizes + patchifies itself
  db.run(`INSERT INTO emb_gemma4 (image_id, v) VALUES (?, ?)`, [id, new Uint8Array(Float32Array.from(vec).buffer)]);
  if (++n % 25 === 0) {
    console.log(`  gemma4 ${n}/${missing.values.length}`);
    fs.writeFileSync(OUT, Buffer.from(db.export()));
  }
}
fs.writeFileSync(OUT, Buffer.from(db.export()));
console.log(`gemma4: done (${(fs.statSync(OUT).size / 1048576).toFixed(1)} MB)`);
