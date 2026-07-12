# Findings: Image Embedding Transformation Invariance

## Method

15 images × 3 encoders (CLIP ViT-B/32, SigLIP B/16, DINOv2-S) × 20 transforms (rotations, crops,
scales, flips, translate, grayscale, brightness, blur, occlusion, combined). For each transform
we take the cosine similarity between its embedding and the original's.

Three things make these numbers trustworthy where an earlier version's were not:

1. **Embeddings are mean-pooled, then L2-normalized.** `image-feature-extraction` returns
   different shapes per model — CLIP `[1, 512]` (already pooled) but SigLIP `[1, 196, 768]` and
   DINOv2 `[1, 257, 384]` (raw, unpooled patch-token grids). Comparing a flattened patch grid
   means a rotation just moves each patch to a different index and cosine collapses — and more
   patches (smaller patch) collapses harder. That single asymmetry produced the earlier
   "patch size is the dominant factor" conclusion. It was an artifact. Pooling to one vector per
   image fixes it.
2. **Every transform is embedded twice** — from raw pixels and after a JPEG round-trip (q0.9) —
   so compression noise is separated from the transform's real effect.
3. **A different-image baseline "floor"** (cosine between unrelated images) is reported per
   model, because a raw similarity is meaningless without knowing how low "unrelated" is.

## Results (raw-pixel mean % across 15 images)

| Transform | CLIP (B/32) | SigLIP (B/16) | DINOv2 (S) |
|---|---|---|---|
| Rotation 90/180/270 | 85 | 85 | 72 |
| Rotation 15/45 | 84 | 88 | 79 |
| Crop 20/35 | 81 | 83 | 81 |
| Scale 80/120 | 88 | 93 | 93 |
| Flip horizontal | 95 | 94 | 97 |
| Flip vertical | 86 | 85 | 81 |
| Grayscale | 91 | 96 | 97 |
| Brightness ±30% | 94 | 97 | 99 |
| Blur | 82 | 80 | 85 |
| Occlude 25% | 93 | 96 | 96 |
| **Different-image floor** | **58** | **52** | **15** |

## Key findings (these replace the earlier report)

1. **"Patch size is the dominant factor" was an artifact — it is gone.** With embeddings pooled,
   SigLIP (patch-16) matches CLIP (patch-32) on rotation (both 85%), and DINOv2 (patch-14) is
   72% — not the ~30–39% the earlier run reported. Patch size does not rank the encoders.

2. **The baseline changes the whole comparison.** The floors are wildly different: CLIP 58%,
   SigLIP 52%, DINOv2 15%. So CLIP's high absolute scores are partly just a compressed embedding
   space — two *unrelated* images already sit at 58%. Normalize each score to its own floor,
   `(sim − floor) / (100 − floor)`, and rotation robustness is:
   - CLIP: (85−58)/42 = **64%**
   - SigLIP: (85−52)/48 = **69%**
   - DINOv2: (72−15)/85 = **67%**
   All three are comparable. The earlier "CLIP is the most transformation-invariant encoder"
   claim does not survive once you account for how spread out each model's space is. If anything,
   DINOv2's low floor means its similarities carry more information per point.

3. **JPEG compression is negligible at q0.9.** Mean |raw − jpeg| across all transforms is 1.4
   points for CLIP and 0.4 for SigLIP/DINOv2. Compression is not what these numbers measure —
   but now that's shown, not assumed. (The earlier browser embedded JPEG-encoded canvases while
   the CLI used raw pixels; that mismatch is removed — both now do both.)

4. **Rotation is still the hardest geometric transform, but not catastrophic.** 72–85% raw,
   comfortably above every floor. The ViT positional-embedding intuition (a 90° turn is a
   genuinely different input) holds directionally — it just costs ~15–28 points, not ~60.

5. **Photometric transforms are nearly free.** Grayscale, brightness, occlusion, and blur all sit
   in the high 80s–90s for every model. These encoders care about content layout far more than
   exact pixels.

## What still holds from the original write-up
- No encoder is *fully* transformation-invariant; a rotated image is not treated as identical.
- Image encoders lack the paraphrase signal text encoders get, so there's no learned
  "a rotated image means the same thing." That framing is right; only the magnitudes were wrong.

## Honest caveats
- Rotation/scale are rendered on a mid-gray square (gray ≈ 0 after normalization, so the
  letterbox contributes little), but a non-square image still changes its image-vs-background
  ratio under rotation. Some residual confound remains.
- Comparing patch size *per se* still needs a within-family ablation (CLIP-B/16 vs B/32); three
  models differ on training, resolution, and objective at once.
- Gemma 4 is available in the browser lab (ONNX Runtime) but not in this CLI batch.

Reproduce: `node cli.mjs test-images/* --model clip|siglip|dinov2 --json`. Interactive:
https://paulkinlan.github.io/image-embedding-lab/ · report: report.html
