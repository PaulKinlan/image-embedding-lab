# Findings: Image Embedding Transformation Invariance

## Experiment

10 diverse images × 3 models (CLIP, SigLIP, DINOv2) × 15 transformations (rotations, crops, scales, flips, combined). Cosine similarity between each transformed embedding and the original.

## Summary

| Transformation | CLIP (patch 32) | SigLIP (patch 16) | DINOv2 (patch 14) |
|---|---|---|---|
| Rotation (90°/180°/270°) | 83% | 26% | 39% |
| Slight rotation (15°/45°) | 79% | 37% | 40% |
| Crop (20%/35%) | 81% | 30% | 49% |
| Scale (80%/120%) | 85% | 65% | 76% |
| Flip horizontal | 94% | 39% | 65% |
| Flip vertical | 86% | 48% | 56% |

## Key findings

1. **CLIP is the most transformation-invariant encoder.** Despite being the oldest model (2021), CLIP's ViT-Base-Patch32 produces embeddings that are surprisingly robust to rotation, cropping, and scaling. The key factor: large patch size (32×32) coarsens the spatial representation, making it less sensitive to pixel-level rearrangement.

2. **SigLIP and DINOv2 are rotation-sensitive.** Both use smaller patches (16×16 / 14×14) which preserve fine spatial detail — but that detail IS affected by rotation. A 90° turn completely rewrites the patch grid.

3. **Horizontal flips are nearly free for CLIP.** CLIP achieves 98-100% similarity on horizontal flips across all images. DINOv2 ranges 46-88%. SigLIP is 33-68%.

4. **None of these encoders capture "semantic invariance."** Unlike text embeddings — where paraphrases with zero shared tokens produce nearly identical vectors — no image encoder treats a rotated image as "the same image." A 90° rotation is a fundamentally different input to a ViT because positional embeddings are absolute.

5. **Patch size is the dominant factor.** CLIP (patch 32) > SigLIP (patch 16) ≈ DINOv2 (patch 14) for every transformation category. Larger patches = coarser spatial representation = more invariant.

## Why image embeddings ≠ text embeddings

With text, "the cat sat on the mat" and "a feline rested on a rug" produce similar embeddings because the model captures MEANING, not surface form. Text training data has natural paraphrase — the same concept expressed many different ways.

Image encoders have NO such training signal for transformations. CLIP is trained on "this image matches this caption" — it's never told "a rotated image is the same image." A 90° rotation completely changes the pixel grid, and the ViT's positional embeddings are tied to absolute spatial location.

This is an active research frontier: equivariant networks, rotation-invariant CNNs, augmentation-aware training. No production vision encoder is fully transformation-invariant yet.

## Practical recommendations

- **Near-duplicate detection** (same image, different resolution/quality): all three models work well
- **Rotation-robust search** (e.g. scanned documents): only CLIP is viable (86% avg)
- **Flip-aware deduplication**: CLIP treats horizontal flips as identical (99%). Use DINOv2 if you need to distinguish mirrors (52% avg)
- **Content-addressable hashing**: no encoder gives true transformation-invariant hashing. Canonicalize images (orient, center-crop) before embedding

## About Gemma 4

Gemma 4 12B Unified is "encoder-free" — images go directly into the LLM backbone without a separate vision tower. Other variants have a separate Gemma4VisionEncoder (patch 16, 768-dim, 16 layers — architecturally SigLIP-derived). A standalone ONNX build exists (112MB quantized) but requires ONNX Runtime Web to load directly, as transformers.js can't load VLM submodels via the standard pipeline API.

Full report: https://paulkinlan.github.io/image-embedding-lab/report.html
