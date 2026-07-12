#!/usr/bin/env node
/** Build tiling-report.html from tiling-results.json. Run after scripts/tiling-experiment.mjs. */
import { readFileSync, writeFileSync } from "node:fs";

const ROOT = new URL("../", import.meta.url).pathname;
const data = JSON.parse(readFileSync(ROOT + "tiling-results.json", "utf8"));
const MODEL_LABELS = { clip: "CLIP (OpenAI)", siglip: "SigLIP (Google)", dinov2: "DINOv2 (Meta)" };
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const col = (p) => (p >= 90 ? "var(--good)" : p >= 70 ? "var(--warn)" : "var(--bad)");

function bars(model) {
  const t = data.models[model];
  const rows = [1, 2, 3].map((g) => {
    const r = t[g];
    if (!r) return "";
    const cells = [
      ["floor (all)", r.floorAll],
      ["floor (detail)", r.floorDetail],
      ["floor (photo)", r.floorPhoto],
      ["rotation", r.rot],
      ["crop", r.crop],
      ["flip H", r.flip],
    ].map(([, v]) => `<td class="num">${pct(v)}</td>`).join("");
    return `<tr><td>${g}×${g}</td>${cells}</tr>`;
  }).join("");
  // narrative: did the DETAIL floor drop from 1×1 to 3×3? (lower floor = more distinguishable)
  const d1 = t[1]?.floorDetail, d3 = t[3]?.floorDetail;
  const drop = d1 != null && d3 != null ? Math.round((d1 - d3) * 100) : null;
  const verdict = drop == null
    ? ""
    : drop > 3
    ? `<div class="finding"><b>${MODEL_LABELS[model]}: tiling helped.</b> The detail-image floor dropped ${drop} points (${pct(d1)} → ${pct(d3)}) from 1×1 to 3×3 — with more tiles the model tells documents apart better, because each crop is seen at full resolution.</div>`
    : drop < -3
    ? `<div class="finding"><b>${MODEL_LABELS[model]}: tiling raised the floor ${-drop} pts.</b> Pooling many tiles averaged the representations together, making images look <i>more</i> alike here.</div>`
    : `<div class="finding"><b>${MODEL_LABELS[model]}: little change (${drop >= 0 ? "-" : "+"}${Math.abs(drop)} pts).</b> Tiling didn't move the detail floor much for this encoder.</div>`;
  return `<h2>${MODEL_LABELS[model]}</h2>
    <table><thead><tr><th>Tiling</th><th class="num">Floor · all</th><th class="num">Floor · detail</th><th class="num">Floor · photo</th><th class="num">Rotation</th><th class="num">Crop</th><th class="num">Flip H</th></tr></thead>
    <tbody>${rows}</tbody></table>${verdict}`;
}

const sections = Object.keys(data.models).map(bars).join("\n");

const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Tiling report — Image Embedding Lab</title>
<style>
:root{color-scheme:light dark;--color:#000;--background:#fdfcf8;--bg-secondary:#f0eee6;--border:#e0ddd4;--muted:#666;--accent:#4b3aff;--good:#1a8a3a;--bad:#c0392b;--warn:#e6a700}
@media(prefers-color-scheme:dark){:root{--color:#e8e4dc;--background:#1c1a17;--bg-secondary:#2a2723;--border:#3a3530;--muted:#9a9088;--accent:#8ab4f8;--good:#57c97a;--bad:#e06c75;--warn:#d9a441}}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;line-height:1.7;color:var(--color);background:var(--background);padding:1.5rem;max-width:820px;margin:0 auto}
h1{font-family:Georgia,serif;font-size:1.8rem;font-weight:normal;margin-bottom:.3rem}
h2{font-family:Georgia,serif;font-size:1.15rem;font-weight:normal;margin:1.6rem 0 .5rem;border-bottom:1px solid var(--border);padding-bottom:.3rem}
.sub{color:var(--muted);font-size:.85rem;margin-bottom:1rem}
a{color:var(--accent)}.back{font-size:.8rem;display:inline-block;margin-bottom:1rem}
table{width:100%;border-collapse:collapse;margin:.8rem 0}
th,td{padding:.45rem .6rem;text-align:left;font-size:.8rem;border-bottom:1px solid var(--border)}
th{background:var(--bg-secondary);font-size:.66rem;text-transform:uppercase;letter-spacing:.03em;color:var(--muted)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.finding{font-size:.85rem;padding:.8rem;background:var(--bg-secondary);border-left:3px solid var(--accent);border-radius:0 .4rem .4rem 0;margin:.6rem 0}
.finding b{color:var(--accent)}
.note{font-size:.75rem;color:var(--muted);margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--border);line-height:1.6}
</style></head><body>
<a class="back" href="index.html">← Back to Image Embedding Lab</a>
<h1>Tiling: does the encoder see more detail?</h1>
<p class="sub">${data.images} images (${data.detail} "detail": webpages, text, charts, diagrams, maps, signs, logos) × 3 encoders × 3 tiling levels. AnyRes tiling splits each image into a grid of crops, embeds each at full model resolution, and mean-pools.</p>
<div class="finding">
<b>The hypothesis (which turned out wrong — the interesting part).</b> A 224px encoder can't read text: a whole webpage is squished to 224, so it just sees "a document". I expected that tiling — showing each crop at full resolution — would let the model tell documents apart, i.e. <b>lower the detail-image floor</b>. The data says the opposite: the floor <b>rises</b> with tiling.
</div>
<div class="finding">
<b>Why it rises: mean-pooling averages the detail away.</b> This lab pools the tile embeddings into one fixed-length vector (so cosine stays comparable). Averaging 4 or 9 local crops pulls every image toward a bland mean, so unrelated images look <i>more</i> alike — the floor goes up, and rotation similarity goes up too (rotations get smoothed together). Real AnyRes VLMs don't do this: they <b>concatenate</b> all the tile tokens and feed the lot to the LLM, keeping the detail. So the honest takeaway is about the <i>combining</i> step, not tiling itself — mean-pool tiling trades discrimination for smoothness. Concatenation is the next thing to try.
</div>
${sections}
<div class="note">
<strong>How to read it.</strong> "Floor" = mean cosine between unrelated images' originals — a discrimination floor, not invariance. Lower is better discrimination. "Detail" vs "photo" splits the floor by whether the image is text/diagram-heavy. Rotation/crop/flip are invariance means (higher = more invariant). Tiling changes both, and not always in the same direction — pooling many tiles can also average away distinctions. Reproduce: <code>node scripts/tiling-experiment.mjs > tiling-results.json &amp;&amp; node scripts/generate-tiling-report.mjs</code>.
</div>
</body></html>`;

writeFileSync(ROOT + "tiling-report.html", html);
console.log("Wrote tiling-report.html");
for (const m of Object.keys(data.models)) {
  const t = data.models[m];
  console.log(`  ${m}: detail floor ${pct(t[1]?.floorDetail)} → ${pct(t[3]?.floorDetail)} (1×1→3×3)`);
}
