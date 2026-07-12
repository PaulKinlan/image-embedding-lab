# Image Embedding Lab

A hands-on lab for building intuition about what image embeddings actually capture. Upload an
image, transform it (rotate, crop, scale, flip, compress…), and watch how much the embedding
moves. Everything runs in your browser via Transformers.js (and there's a Node CLI for batch
runs) — no server, no API key.

Live: https://paulkinlan.github.io/image-embedding-lab/ · [VLM playground](vlm.html) ·
[Batch report](report.html) · [Tiling report](tiling-report.html) · [Findings](FINDINGS.md)

The **VLM playground** (`vlm.html`) runs Florence-2 in the browser — caption an image, read its
text (OCR), or detect objects. It's the counterpart to the embedding lab: where the lab reduces
an image to one vector, a VLM turns it into text, so you can watch it read a page an embedding
model can only see as "a document".

## What this is really about

The starting question was simple: **are image embeddings invariant to transformations?** If you
rotate or crop an image, does a vision encoder still produce "the same" vector? Text embeddings
famously do this for *meaning* — "the cat sat on the mat" and "a feline rested on a rug" land in
nearly the same place. Do image encoders treat a rotated photo as the same photo?

Answering it turned into a tour of how these models — and the way you measure them —
actually work. This README is the running log of the questions and what each one taught me.

## The questions I've been chasing

**1. Are the embeddings transformation-invariant?**
Partly. Flips and photometric changes (brightness, grayscale, blur, occlusion) barely move the
embedding; rotation moves it the most. But "how much is a lot?" turned out to depend entirely on
the next two questions.

**2. Was I even measuring the right thing? (the big one)**
No — at first. `image-feature-extraction` returns different shapes per model: CLIP gives a pooled
`[1, 512]` vector, but SigLIP returns `[1, 196, 768]` and DINOv2 `[1, 257, 384]` — the *raw,
unpooled grid of patch tokens*. Comparing a flattened patch grid means a rotation just shuffles
which patch is at which index, so cosine collapses — and smaller patches (more tokens) collapse
harder. That single mistake manufactured a confident-but-wrong headline ("patch size is the
dominant factor"). The fix: **mean-pool to one vector per image** before comparing. After that,
SigLIP ties CLIP on rotation and the "patch size" story evaporates. Lesson: with embeddings,
*how you pool is part of the model*.

**3. How low is "low"? (the baseline floor)**
A 70% cosine sounds high — but high compared to what? So the lab now measures the **different-
image floor**: the cosine between *unrelated* images. It varies wildly — CLIP ~58%, DINOv2 ~15%.
That reframes everything: CLIP's high scores are partly just a compressed embedding space where
even random images sit at 58%. Every bar in the UI now has a floor tick; a result only means
"invariant" if it's well to the right of the floor, not just near 100%.

**4. Does JPEG compression change the picture?**
Each transform is embedded at raw pixels **and** at JPEG q0.9 / q0.5 / q0.2, so compression is
its own axis. So far: at q0.9 it's negligible (~1 point); the interesting question is whether
invariance holds as quality drops toward q0.2.

**5. Why is everything squished to 224×224? How do LLMs read text then?**
Because these encoders are *fixed* at 224 — the ViT patch grid and its positional embeddings are
baked in at training, so they physically can't take a bigger image. At 224 a patch is 14–32px,
far too coarse for text. That's why these are *semantic* encoders, not OCR. The big VLMs
(GPT-4V, Claude, Qwen2-VL, InternVL, Gemma 3/4) read text by using higher native resolution *and*
**tiling** — splitting a page into many crops, encoding each, and concatenating thousands of
vision tokens. OCR comes from throwing far more patches at the image, not a smarter low-res pass.

**6. What do the different encoders "see"?**
- **CLIP** (OpenAI, 2021) — trained on image↔caption pairs, so it's language-aligned.
- **SigLIP** (Google) — same idea, sigmoid loss, generally sharper.
- **DINOv2** (Meta) — *self-supervised*, never told about language; a good contrast.
- **Gemma 4 vision** (experimental here, via ONNX Runtime) — a SigLIP-derived encoder pulled out
  of a VLM.
Comparing language-aligned vs self-supervised is a big part of the fun.

**7. Semantic invariance vs surface form.**
Even fixed, no encoder treats a rotated image as *identical* — a ViT's positional embeddings are
absolute, so a turn is a different input. Image encoders lack the paraphrase signal
text encoders get ("this rotated image means the same thing" is never in the training data).
That framing holds; the earlier mistake was only in the magnitudes.

See **[FINDINGS.md](FINDINGS.md)** for the current numbers and **[report.html](report.html)** for
the batch charts.

## Using it

**Browser** — open the live link (or serve locally, below). Pick an encoder (224px, higher-res
384/336, or experimental Gemma 4), choose a **tiling** level (whole image / 2×2 / 3×3), drop an
image, and either drag the sliders + "Test this transform" for a single check, or "Compare all
transforms" for the full sweep. A progress bar shows each embedding as it runs; results show the
raw-pixel bar with the JPEG q90/50/20 values beneath, plus the different-image floor.

Tiling splits the image into crops, embeds each at full model resolution, and pools them — so the
encoder effectively sees 2×/3× the detail. It's the AnyRes trick VLMs use to read text: try a
webpage screenshot at 1×1 vs 3×3 and watch the different-image floor drop as the model starts to
tell documents apart.

**CLI** (batch, Node):

```sh
npm install
node cli.mjs test-images/*.jpg --model clip                 # human-readable table
node cli.mjs test-images/*  --model siglip384               # higher-res encoder
node cli.mjs test-images/*  --model dinov2 --tiles 3         # 3×3 AnyRes tiling
node cli.mjs test-images/*  --model dinov2 --json > results-dinov2.json
python3 scripts/generate_report.py                          # regenerate report.html
```

Models: `clip`, `siglip`, `dinov2` (224px), `siglip384`, `clip-l` (higher-res). `--tiles N` for
AnyRes tiling.

Both share `lib/experiment.mjs` (transforms, pooling, cosine, JPEG variants) so the browser and
CLI measure identically.

**Run locally**

```sh
python3 -m http.server 8000   # then open http://localhost:8000
```

## Open questions / where this is going

- ✅ **Tiling + higher-resolution encoders** — now in: SigLIP-384 / CLIP-L-336 plus a 2×2 / 3×3
  AnyRes tiling mode. Next: measure the floor drop on documents systematically.
- **Isolate patch size properly** — compare within one family (CLIP-B/16 vs B/32) instead of
  across three models that differ on everything.
- **Text↔image alignment** — for CLIP/SigLIP, test whether a transformed image still matches the
  *original's caption*, which is a more meaningful "does it still mean the same thing" test than
  image↔image cosine.
- **Heavier compression / other corruptions** — noise, resize artifacts, screenshots of
  screenshots.

Suggestions and questions welcome — this is a learning project as much as a benchmark.
