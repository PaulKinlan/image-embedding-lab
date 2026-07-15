// Shared experiment core for Image Embedding Lab.
//
// Pure logic used by BOTH the browser (index.html) and the Node CLI (cli.mjs) so the two
// pipelines can never drift (that drift is exactly what produced the JPEG-vs-raw mismatch).
// Nothing here imports a model runtime — callers pass in an `embed(pixels|canvas)` function.

export const SIZE = 224;

// JPEG quality levels tested alongside raw pixels. Each transform is embedded at raw + each of
// these, so you can see whether transform-invariance survives increasingly heavy compression.
export const VARIANTS = [
  { key: "raw", quality: null, label: "raw" },
  { key: "q90", quality: 0.9, label: "jpeg 90" },
  { key: "q50", quality: 0.5, label: "jpeg 50" },
  { key: "q20", quality: 0.2, label: "jpeg 20" },
];

// Mid-gray background. After a model's (x - mean) / std normalization, ~0.5 maps close to zero,
// so letterbox padding contributes far less to the embedding than white (1.0) did. This blunts
// the "rotation similarity is really measuring the white border" confound.
export const BG = "#7f7f7f";

// Transform set: geometric params + optional pixel post-effects. Post-effects are applied to
// the rendered ImageData, so they behave identically under node-canvas and the browser.
export const TRANSFORMS = [
  { name: "Identity", p: {} },
  { name: "Rotate 90°", p: { rotation: 90 } },
  { name: "Rotate 180°", p: { rotation: 180 } },
  { name: "Rotate 270°", p: { rotation: 270 } },
  { name: "Rotate 15°", p: { rotation: 15 } },
  { name: "Rotate 45°", p: { rotation: 45 } },
  { name: "Crop 10%", p: { crop: 0.10 } },
  { name: "Crop 20%", p: { crop: 0.20 } },
  { name: "Crop 35%", p: { crop: 0.35 } },
  { name: "Scale 80%", p: { scale: 0.8 } },
  { name: "Scale 120%", p: { scale: 1.2 } },
  { name: "Flip H", p: { flip: "horizontal" } },
  { name: "Flip V", p: { flip: "vertical" } },
  { name: "Translate 15%", p: { tx: 0.15, ty: 0.10 } },
  { name: "Grayscale", p: { gray: true } },
  { name: "Brighten 1.3×", p: { brightness: 1.3 } },
  { name: "Darken 0.7×", p: { brightness: 0.7 } },
  { name: "Blur", p: { blur: 3 } },
  { name: "Occlude 25%", p: { occlude: 0.25 } },
  { name: "Rotate 90° + Crop 20%", p: { rotation: 90, crop: 0.20 } },
];

export function imgDims(img) {
  return { w: img.naturalWidth || img.width, h: img.naturalHeight || img.height };
}

/**
 * Pure-JS area-average downscale of an RGBA buffer into a size×size square with aspect-fit and
 * the lab's mid-gray letterbox. Exists because canvas downscaling is runtime-specific (Skia in
 * Chrome, cairo in node-canvas produce different pixels for the same drawImage), which shifts
 * embeddings by whole points of cosine. Render at NATIVE resolution first (scale factor 1, so
 * the runtime's resampler never runs), then downscale with this shared code — identical bytes
 * in every runtime. Box filter with exact fractional coverage; bilinear when upscaling.
 */
export function fitToSquare(src, size, bg = 127) {
  const scale = Math.min(size / src.width, size / src.height);
  const dw = Math.max(1, Math.round(src.width * scale));
  const dh = Math.max(1, Math.round(src.height * scale));
  const ox = Math.floor((size - dw) / 2), oy = Math.floor((size - dh) / 2);
  const out = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < out.length; i += 4) { out[i] = bg; out[i + 1] = bg; out[i + 2] = bg; out[i + 3] = 255; }
  const sx = src.width / dw, sy = src.height / dh;
  const d = src.data;
  for (let y = 0; y < dh; y++) {
    const y0 = y * sy, y1 = Math.min(src.height, (y + 1) * sy);
    for (let x = 0; x < dw; x++) {
      const x0 = x * sx, x1 = Math.min(src.width, (x + 1) * sx);
      let r = 0, g = 0, b = 0, wsum = 0;
      if (sx >= 1 || sy >= 1) {
        // downscale: exact fractional box coverage
        for (let yy = Math.floor(y0); yy < y1; yy++) {
          const wy = Math.min(yy + 1, y1) - Math.max(yy, y0);
          if (wy <= 0) continue;
          for (let xx = Math.floor(x0); xx < x1; xx++) {
            const wx = Math.min(xx + 1, x1) - Math.max(xx, x0);
            if (wx <= 0) continue;
            const o = (yy * src.width + xx) * 4;
            const w = wx * wy;
            r += d[o] * w; g += d[o + 1] * w; b += d[o + 2] * w;
            wsum += w;
          }
        }
      } else {
        // upscale: bilinear sample at the pixel center
        const cx = Math.min(src.width - 1.001, Math.max(0, (x + 0.5) * sx - 0.5));
        const cy = Math.min(src.height - 1.001, Math.max(0, (y + 0.5) * sy - 0.5));
        const ix = Math.floor(cx), iy = Math.floor(cy);
        const fx = cx - ix, fy = cy - iy;
        for (const [ddx, ddy, w] of [[0, 0, (1 - fx) * (1 - fy)], [1, 0, fx * (1 - fy)], [0, 1, (1 - fx) * fy], [1, 1, fx * fy]]) {
          const o = ((iy + ddy) * src.width + (ix + ddx)) * 4;
          r += d[o] * w; g += d[o + 1] * w; b += d[o + 2] * w;
          wsum += w;
        }
      }
      const o = ((oy + y) * size + (ox + x)) * 4;
      out[o] = r / wsum; out[o + 1] = g / wsum; out[o + 2] = b / wsum; out[o + 3] = 255;
    }
  }
  return { data: out, width: size, height: size };
}

// Tile input size. A grid of tiles is rendered at grid*TILE, so each tile is a full TILE-px crop
// of the source — i.e. tiling multiplies the effective resolution the encoder sees (this is how
// real VLMs get OCR-level detail out of fixed-resolution vision encoders: many crops, not one
// downscaled pass). At grid=1 it's a plain whole-image embed.
export const TILE = 224;

/**
 * Split a source canvas into grid×grid tiles, each drawn into its own TILE×TILE canvas via the
 * given createCanvas(w,h) factory (node-canvas in Node, document.createElement in the browser).
 * grid<=1 returns the source unchanged (as a single "tile").
 */
export function makeTiles(source, grid, createCanvas, tileSize = TILE) {
  if (grid <= 1) return [source];
  const tiles = [];
  const tw = source.width / grid, th = source.height / grid;
  for (let gy = 0; gy < grid; gy++) {
    for (let gx = 0; gx < grid; gx++) {
      const c = createCanvas(tileSize, tileSize);
      c.getContext("2d").drawImage(source, gx * tw, gy * th, tw, th, 0, 0, tileSize, tileSize);
      tiles.push(c);
    }
  }
  return tiles;
}

/** Average a set of (already-pooled) vectors into one, L2-normalized. Used to combine tile
 *  embeddings into a single comparable vector. */
export function meanPoolVectors(vectors) {
  const dim = vectors[0].length;
  const out = new Float64Array(dim);
  for (const v of vectors) for (let i = 0; i < dim; i++) out[i] += v[i];
  for (let i = 0; i < dim; i++) out[i] /= vectors.length;
  let mag = 0;
  for (let i = 0; i < dim; i++) mag += out[i] * out[i];
  mag = Math.sqrt(mag) || 1;
  for (let i = 0; i < dim; i++) out[i] /= mag;
  return out;
}

// Render a transform of `img` into the 2D context of a `size`×`size` canvas.
export function renderTransform(ctx, img, p = {}, size = SIZE) {
  ctx.save();
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, size, size);

  const { w: iw, h: ih } = imgDims(img);
  const crop = p.crop || 0;
  const scale = p.scale || 1;
  const sw = iw * (1 - crop * 2);
  const sh = ih * (1 - crop * 2);
  const sx = iw * crop;
  const sy = ih * crop;
  const aspect = sw / sh;
  let dw = size * scale, dh = size * scale;
  if (aspect > 1) dh = dw / aspect;
  else dw = dh * aspect;

  ctx.translate(size / 2, size / 2);
  if (p.tx || p.ty) ctx.translate((p.tx || 0) * size, (p.ty || 0) * size);
  if (p.rotation) ctx.rotate((p.rotation * Math.PI) / 180);
  if (p.flip === "horizontal") ctx.scale(-1, 1);
  if (p.flip === "vertical") ctx.scale(1, -1);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(img, sx, sy, sw, sh, -dw / 2, -dh / 2, dw, dh);
  ctx.restore();

  if (p.gray || p.brightness || p.blur || p.occlude || p.shuffle) applyPixelEffects(ctx, size, p);
}

function applyPixelEffects(ctx, size, p) {
  const im = ctx.getImageData(0, 0, size, size);
  const d = im.data;
  if (p.gray) {
    for (let i = 0; i < d.length; i += 4) {
      const y = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      d[i] = d[i + 1] = d[i + 2] = y;
    }
  }
  if (p.brightness) {
    const f = p.brightness;
    for (let i = 0; i < d.length; i += 4) {
      d[i] = Math.min(255, d[i] * f);
      d[i + 1] = Math.min(255, d[i + 1] * f);
      d[i + 2] = Math.min(255, d[i + 2] * f);
    }
  }
  if (p.blur) boxBlur(d, size, size, p.blur);
  if (p.occlude) {
    const s = Math.round(size * p.occlude);
    const x0 = Math.round((size - s) / 2), y0 = Math.round((size - s) / 2);
    for (let y = y0; y < y0 + s; y++) {
      for (let x = x0; x < x0 + s; x++) {
        const i = (y * size + x) * 4;
        d[i] = d[i + 1] = d[i + 2] = 127;
      }
    }
  }
  // Patch shuffle: permute a g×g grid of tiles with a SEEDED Fisher–Yates, so every image and
  // every model sees the exact same permutation. Content is preserved pixel-for-pixel; only
  // spatial arrangement is destroyed — the cleanest probe for "does the embedding encode
  // what-is-where or just what". Uses floor(size/g) tiles anchored top-left (exact for the
  // sizes we use: 224 and 768 are divisible by 2, 4, and 8).
  if (p.shuffle && p.shuffle >= 2) {
    const g = Math.floor(p.shuffle);
    const t = Math.floor(size / g);
    const src = new Uint8ClampedArray(d);   // snapshot before writing
    let s = (p.shuffleSeed ?? 42) >>> 0;
    const rand = () => {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let x = Math.imul(s ^ (s >>> 15), 1 | s);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
    const perm = Array.from({ length: g * g }, (_, i) => i);
    for (let i = perm.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [perm[i], perm[j]] = [perm[j], perm[i]];
    }
    for (let cell = 0; cell < g * g; cell++) {
      const from = perm[cell];
      const dx = (cell % g) * t, dy = Math.floor(cell / g) * t;
      const sx2 = (from % g) * t, sy2 = Math.floor(from / g) * t;
      for (let row = 0; row < t; row++) {
        const so = ((sy2 + row) * size + sx2) * 4;
        const do2 = ((dy + row) * size + dx) * 4;
        d.set(src.subarray(so, so + t * 4), do2);
      }
    }
  }
  ctx.putImageData(im, 0, 0);
}

// Simple separable box blur (two passes) over RGBA pixel data. Cross-environment, no ctx.filter.
function boxBlur(d, w, h, r) {
  const pass = (src, horizontal) => {
    const out = new Uint8ClampedArray(src.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let rs = 0, gs = 0, bs = 0, n = 0;
        for (let k = -r; k <= r; k++) {
          const xx = horizontal ? x + k : x;
          const yy = horizontal ? y : y + k;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const i = (yy * w + xx) * 4;
          rs += src[i];
          gs += src[i + 1];
          bs += src[i + 2];
          n++;
        }
        const i = (y * w + x) * 4;
        out[i] = rs / n;
        out[i + 1] = gs / n;
        out[i + 2] = bs / n;
        out[i + 3] = src[i + 3];
      }
    }
    return out;
  };
  const h1 = pass(d, true);
  const v = pass(h1, false);
  d.set(v);
}

export function cosineSim(a, b) {
  let dot = 0, ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    ma += a[i] * a[i];
    mb += b[i] * b[i];
  }
  return dot / (Math.sqrt(ma) * Math.sqrt(mb) || 1);
}

// Pool a raw feature-extraction output into ONE comparable, L2-normalized vector.
//   dims [1, D]     -> the model already pooled; pass through.
//   dims [1, T, D]  -> mean-pool over the T tokens (the fix: without this, SigLIP/DINOv2 are
//                      compared as flattened spatially-ordered patch grids, so rotation trivially
//                      scrambles them and manufactures a fake "patch size" effect).
export function poolEmbedding(data, dims) {
  const nd = dims.length;
  const embedDim = dims[nd - 1];
  const numTokens = nd >= 3 ? dims[nd - 2] : 1;
  const out = new Float64Array(embedDim);
  if (numTokens === 1) {
    for (let i = 0; i < embedDim; i++) out[i] = data[i];
  } else {
    for (let t = 0; t < numTokens; t++) {
      for (let e = 0; e < embedDim; e++) out[e] += data[t * embedDim + e];
    }
    for (let e = 0; e < embedDim; e++) out[e] /= numTokens;
  }
  let mag = 0;
  for (let i = 0; i < embedDim; i++) mag += out[i] * out[i];
  mag = Math.sqrt(mag) || 1;
  for (let i = 0; i < embedDim; i++) out[i] /= mag;
  return out;
}

// Mean and sample standard deviation of an array (for cross-image stats).
export function meanStd(xs) {
  const v = xs.filter((x) => x != null && !Number.isNaN(x));
  if (!v.length) return { mean: null, std: null, n: 0 };
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const varr = v.length > 1 ? v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1) : 0;
  return { mean, std: Math.sqrt(varr), n: v.length };
}
