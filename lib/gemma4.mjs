// Gemma 4 vision-encoder preprocessing — a faithful JS port of transformers'
// Gemma4ImageProcessor (patch_size 16, max_soft_tokens 280, pooling 3, rescale 1/255, no
// normalization, channel-last patches, (col,row) position ids). The ONNX "vision_encoder"
// takes PRE-PATCHIFIED input: pixel_values [1, seq, 768] + pixel_position_ids [1, seq, 2] —
// feeding it a raw image can never work, which is why the first Gemma 4 attempt failed.
// Runs in Node and the browser; the caller supplies an RGBA buffer and an ONNX session.

export const GEMMA4 = {
  patchSize: 16,
  maxSoftTokens: 280,
  poolingKernel: 3,
};

/** Target (height, width): largest dims that fit the patch budget and divide by 48. */
export function gemma4TargetSize(height, width, { patchSize = 16, maxSoftTokens = 280, poolingKernel = 3 } = {}) {
  const maxPatches = maxSoftTokens * poolingKernel * poolingKernel;
  const targetPx = maxPatches * patchSize * patchSize;
  const factor = Math.sqrt(targetPx / (height * width));
  const sideMult = poolingKernel * patchSize;
  let th = Math.floor((factor * height) / sideMult) * sideMult;
  let tw = Math.floor((factor * width) / sideMult) * sideMult;
  const maxSide = Math.floor(maxPatches / (poolingKernel * poolingKernel)) * sideMult;
  if (th === 0 && tw === 0) throw new Error("image too small for gemma4 preprocessing");
  if (th === 0) { th = sideMult; tw = Math.min(Math.floor(width / height) * sideMult, maxSide); }
  else if (tw === 0) { tw = sideMult; th = Math.min(Math.floor(height / width) * sideMult, maxSide); }
  return { height: th, width: tw };
}

/** Bilinear resize of an RGBA buffer ({data,width,height}) to exact w×h (shared across runtimes). */
export function resizeBilinear(src, dstW, dstH) {
  const out = new Uint8ClampedArray(dstW * dstH * 4);
  const sx = src.width / dstW, sy = src.height / dstH;
  const d = src.data;
  for (let y = 0; y < dstH; y++) {
    const cy = Math.min(src.height - 1.001, Math.max(0, (y + 0.5) * sy - 0.5));
    const iy = Math.floor(cy), fy = cy - iy;
    for (let x = 0; x < dstW; x++) {
      const cx = Math.min(src.width - 1.001, Math.max(0, (x + 0.5) * sx - 0.5));
      const ix = Math.floor(cx), fx = cx - ix;
      const o = (y * dstW + x) * 4;
      for (let ch = 0; ch < 4; ch++) {
        const v =
          d[(iy * src.width + ix) * 4 + ch] * (1 - fx) * (1 - fy) +
          d[(iy * src.width + ix + 1) * 4 + ch] * fx * (1 - fy) +
          d[((iy + 1) * src.width + ix) * 4 + ch] * (1 - fx) * fy +
          d[((iy + 1) * src.width + ix + 1) * 4 + ch] * fx * fy;
        out[o + ch] = v;
      }
    }
  }
  return { data: out, width: dstW, height: dstH };
}

/**
 * RGBA buffer → { pixelValues: Float32Array [seq*768], positionIds: BigInt64Array [seq*2], seq }.
 * Patch values are channel-last within each patch ([py][px][c]), rescaled to [0,1];
 * position ids are (col,row) per patch, row-major over the patch grid.
 */
export function gemma4Patchify(rgba, { patchSize = 16 } = {}) {
  const { width, height, data } = rgba;
  const nW = Math.floor(width / patchSize), nH = Math.floor(height / patchSize);
  const seq = nW * nH;
  const per = patchSize * patchSize * 3;
  const pixelValues = new Float32Array(seq * per);
  const positionIds = new BigInt64Array(seq * 2);
  for (let gy = 0; gy < nH; gy++) {
    for (let gx = 0; gx < nW; gx++) {
      const p = gy * nW + gx;
      positionIds[p * 2] = BigInt(gx);       // x (col)
      positionIds[p * 2 + 1] = BigInt(gy);   // y (row)
      let k = p * per;
      for (let py = 0; py < patchSize; py++) {
        const row = (gy * patchSize + py) * width;
        for (let px = 0; px < patchSize; px++) {
          const o = (row + gx * patchSize + px) * 4;
          pixelValues[k++] = data[o] / 255;
          pixelValues[k++] = data[o + 1] / 255;
          pixelValues[k++] = data[o + 2] / 255;
        }
      }
    }
  }
  return { pixelValues, positionIds, seq };
}

/** Full pipeline: RGBA at native size → resized → patchified → session.run → mean-pooled unit vector. */
export async function gemma4Embed(session, ort, rgba) {
  const { height, width } = gemma4TargetSize(rgba.height, rgba.width);
  const resized = resizeBilinear(rgba, width, height);
  const { pixelValues, positionIds, seq } = gemma4Patchify(resized);
  const feeds = {
    pixel_values: new ort.Tensor("float32", pixelValues, [1, seq, pixelValues.length / seq]),
    pixel_position_ids: new ort.Tensor("int64", positionIds, [1, seq, 2]),
  };
  const out = await session.run(feeds);
  const t = out[session.outputNames[0]];
  // image_features is [tokens, dim] (no batch dim); tolerate a leading batch dim anyway
  const dims = t.dims.length === 3 ? t.dims.slice(1) : t.dims;
  const [tokens, dim] = dims;
  const v = new Float64Array(dim);
  for (let tok = 0; tok < tokens; tok++) {
    for (let d2 = 0; d2 < dim; d2++) v[d2] += Number(t.data[tok * dim + d2]);
  }
  let mag = 0;
  for (let d2 = 0; d2 < dim; d2++) { v[d2] /= tokens; mag += v[d2] * v[d2]; }
  mag = Math.sqrt(mag) || 1;
  for (let d2 = 0; d2 < dim; d2++) v[d2] /= mag;
  return v;
}
