# Image Embedding Lab

Are image embeddings invariant to rotation, cropping, and scaling? Upload an image and find out. CLIP's vision encoder runs entirely in your browser via Transformers.js — no server, no API key.

## How it works

1. Load the CLIP ViT-Base vision encoder (~100MB, one-time download) via Transformers.js
2. Upload an image
3. The lab computes embeddings for the original and 12 transformed versions (rotations, crops, scales, flips)
4. Cosine similarity between each transform and the original tells you how invariant the encoder is

## Run locally

Just open index.html — it's a single static page. Or serve it:

```sh
python3 -m http.server 8000
```

Live: https://paulkinlan.github.io/image-embedding-lab/
