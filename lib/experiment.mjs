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

  if (p.gray || p.brightness || p.blur || p.occlude) applyPixelEffects(ctx, size, p);
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
