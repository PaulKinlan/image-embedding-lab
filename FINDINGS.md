# Findings: Image Embedding Transformation Invariance

## Method

15 images × N encoders (CLIP ViT-B/32, SigLIP B/16, DINOv2-S, plus higher-res SigLIP-384 /
CLIP-L-336) × 20 transforms (rotations, crops, scales, flips, translate, grayscale, brightness,
blur, occlusion, combined). For each transform we take the cosine similarity between its
embedding and the original's.

### The embedding method

An embedding here is one vector per image, produced the same way for every encoder:

1. **Render → encode → pool → normalize.** The image is drawn to a square canvas, passed to the
   vision encoder, reduced to ONE vector, and L2-normalized. That vector is what every number
   below compares.
2. **Pooling is the load-bearing step.** `image-feature-extraction` returns different shapes per
   model — CLIP `[1, 512]` (already pooled) but SigLIP `[1, 196, 768]` and DINOv2 `[1, 257, 384]`
   (raw, unpooled patch-token grids). Comparing a flattened patch grid means a rotation moves
   each patch to a different index and cosine collapses — more patches collapse harder. That
   asymmetry produced the earlier "patch size is the dominant factor" conclusion; it was an
   artifact. Mean-pooling every model's tokens to one vector removes it.
3. **JPEG as an axis.** Every transform is embedded at raw pixels and three JPEG qualities
   (q0.9 / q0.5 / q0.2), to separate compression noise from the transform's real effect.
4. **Tiling (AnyRes).** Optionally the image is split into a grid of crops, each embedded at full
   resolution and mean-pooled — the trick VLMs use to see detail. See the tiling report.
5. **Baseline floor.** The cosine between unrelated images is reported per model, because a raw
   similarity means nothing without knowing how low "unrelated" is.

### Getting an embedding out of a VLM

A vision-language model builds the same object internally: its vision encoder turns the image
into a grid of token vectors, and a language decoder reads them to write text. To get an
embedding you tap that grid and pool it — step 1 above. Florence-2's `encode_image` returns
`[1, 577, 768]`; mean-pool → a 768-d image embedding that compares by cosine like the encoders
here (identical image 100%, kitten vs mountain 67%, two webpages 91%, photo vs webpage 43%). Try
it in the VLM playground (vlm.html). The point: a model works with per-token embeddings at every
layer; "an embedding" is one pooled hidden state, versus running that hidden state through the
output layer to pick the next word.

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
   different input) holds directionally — it costs ~15–28 points, not ~60.

5. **Photometric transforms are nearly free.** Grayscale, brightness, occlusion, and blur all sit
   in the high 80s–90s for every model. These encoders care about content layout far more than
   exact pixels.

## What still holds from the original write-up
- No encoder is *fully* transformation-invariant; a rotated image is not treated as identical.
- Image encoders lack the paraphrase signal text encoders get, so there's no learned
  "a rotated image means the same thing." That framing is right; only the magnitudes were wrong.

## Caveats
- Rotation/scale are rendered on a mid-gray square (gray ≈ 0 after normalization, so the
  letterbox contributes little), but a non-square image still changes its image-vs-background
  ratio under rotation. Some residual confound remains.
- Comparing patch size *per se* still needs a within-family ablation (CLIP-B/16 vs B/32); three
  models differ on training, resolution, and objective at once.
- Gemma 4 is available in the browser lab (ONNX Runtime) but not in this CLI batch.

Reproduce: `node cli.mjs test-images/* --model clip|siglip|dinov2 --json`. Interactive:
https://paulkinlan.github.io/image-embedding-lab/ · report: report.html
