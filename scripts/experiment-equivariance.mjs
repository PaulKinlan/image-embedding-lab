// Experiment 3 — EQUIVARIANCE vs INVARIANCE. Pooled cosine drops under rotation — but did the
// representation BREAK, or did it just MOVE with the image? Compare UNPOOLED token grids:
//   raw          = cos(tokens(rot(img)), tokens(img))                      (flattened grids)
//   compensated  = cos(inverse-rotate-positions(tokens(rot(img))), tokens(img))
// If compensation recovers similarity, the encoder is EQUIVARIANT: each patch token stayed
// faithful, only its position moved. That's layout-preservation in its purest form.
// CLIP's image-feature-extraction output is already pooled/projected ([1,512]) so it has no
// grid here; we test SigLIP (14×14), DINOv2 (CLS+16×16) and Florence (CLS?+24×24).
import fs from "node:fs";
import { MODELS, CORPUS, embed, cosineSim, meanStd, fmt } from "./lib-harness.mjs";

const ANGLES = [90, 180, 270];
const GRID_MODELS = ["siglip", "dinov2", "florence"];

// Split dims [1, T, D] into { cls: n leading non-grid tokens, g } where T - n = g*g.
function gridShape(T) {
  for (let n = 0; n <= 1; n++) {
    const g = Math.sqrt(T - n);
    if (Number.isInteger(g)) return { lead: n, g };
  }
  throw new Error(`token count ${T} is not (CLS+)g²`);
}

// Rotate token grid positions by `angle` CW. tokens: flat Float64Array[T*D].
function rotateGrid(tokens, T, D, lead, g, angle) {
  const out = new Float64Array(T * D);
  out.set(tokens.subarray(0, lead * D));   // leading CLS token(s) stay put
  const steps = ((angle / 90) % 4 + 4) % 4;
  for (let y = 0; y < g; y++) {
    for (let x = 0; x < g; x++) {
      let sx = x, sy = y;
      // PULL mapping: out[p] = in[src] undoes a rotation when src = FORWARD-rotate(p).
      // (Verified against a synthetic 2×2/4×4 grid — the naive "inverse per step" fetches the
      // 180°-off cell.) Forward 90° CW in canvas coords: (col,row) → (g-1-row, col).
      for (let s = 0; s < steps; s++) {
        const nx = g - 1 - sy, ny = sx;
        sx = nx; sy = ny;
      }
      const src = (lead + sy * g + sx) * D;
      const dst = (lead + y * g + x) * D;
      out.set(tokens.subarray(src, src + D), dst);
    }
  }
  return out;
}

const results = {};
for (const model of GRID_MODELS) {
  console.log(`\n=== ${model} ===`);
  const perAngle = {};
  for (const angle of ANGLES) {
    const raw = [], comp = [];
    for (const file of CORPUS) {
      const base = await embed(model, file, {}, { withTokens: true });
      const rot = await embed(model, file, { rotation: angle }, { withTokens: true });
      const T = base.dims[base.dims.length - 2], D = base.dims[base.dims.length - 1];
      const { lead, g } = gridShape(T);
      const bt = Float64Array.from(base.tokens);
      const rt = Float64Array.from(rot.tokens);
      raw.push(cosineSim(bt, rt));
      // The image was rotated CW by `angle`; rotate the token grid positions back (CCW), i.e.
      // apply the inverse rotation to the rotated image's grid before comparing.
      comp.push(cosineSim(bt, rotateGrid(rt, T, D, lead, g, angle)));
    }
    const r = meanStd(raw), c = meanStd(comp);
    perAngle[angle] = { raw: r.mean, compensated: c.mean, recovery: (c.mean - r.mean) / (1 - r.mean) };
    console.log(`${angle}°  raw ${fmt(r.mean)}  compensated ${fmt(c.mean)}  recovery ${fmt((c.mean - r.mean) / (1 - r.mean))}`);
  }
  results[model] = perAngle;
}
fs.writeFileSync("results-equivariance.json", JSON.stringify({ angles: ANGLES, corpus: CORPUS.length, results }, null, 2));
console.log("\nwritten results-equivariance.json");
