# AGENTS.md — Image Embedding Lab

Rules for any agent (or human) working in this repo. Paul uses these results in public
analysis — a wrong number here makes him look wrong in public. Every rule below traces to a
real bug that shipped and had to be found later; the commit that fixed it is cited so you can
read the full story. When in doubt: measure, don't assume, and verify the same number in both
runtimes before pushing.

## Measurement constraints

These have all been violated once. Don't repeat them.

1. **Pool before you compare.** Extract ONE embedding per input: mean-pool tokens + L2-normalize
   (`poolEmbedding` in `lib/experiment.mjs`). Never compare flattened token grids — models
   return different shapes (`[1,512]` pooled vs `[1,196,768]` patch tokens), and comparing a
   pooled vector against a spatially-ordered grid manufactures findings. This produced (and
   inverted) the original "patch size is dominant" headline. (`bb4cb3f`)

2. **Feed each encoder its native input resolution.** CLIP-B/32 and friends: 224 (or 336/384
   for the higher-res variants). Florence-2: 768. Never render to the lab's 224 and let a
   768-native processor upscale it — Chrome's canvas rotate/downscale aliases badly and the
   upscale magnifies the artifacts (~0.25 cosine on rotation, looked like a model weakness).
   Browser canvas filtering ≠ node-canvas (cairo) filtering: identical code can produce
   different pixels per runtime. (`871c403`)

3. **No similarity number without its floor.** Always show the different-image baseline next to
   any cosine. Floors differ hugely per model (CLIP ~58%, DINOv2 ~15% at 224) and do most of
   the explaining. Raw image↔text cosines sit ~0.2 (CLIP modality gap) — only relative
   comparisons mean anything. (`bb4cb3f`, `4e864c1`)

4. **Browser and CLI must be the same pipeline — code, version, pixels, AND precision.** Shared
   logic lives in `lib/experiment.mjs` / `lib/vector-viz.mjs` — never duplicate it into a page
   or the CLI; they drift (the JPEG-vs-raw incident). The pages pin `transformers@X` on the CDN
   and `package.json` resolves a version for Node — keep them THE SAME and bump them together
   (they silently diverged 3.0.0 vs 3.8.1 for months). Canvas resampling is runtime-specific
   (Skia ≠ cairo): render at native scale and downscale with the shared `fitToSquare`. And
   **model dtype must match across runtimes**: browser WASM defaults to q8-quantized ONNX
   while Node defaults to fp32 — cross-precision comparisons cost ~0.14 cosine on identical
   pixels (the search index is built q8 and the page forces q8 on every backend for this
   reason). (`bb4cb3f`, `871c403`, search-demo commits)

5. **One variable at a time.** CLIP-B/32 vs SigLIP-B/16 vs DINOv2-S differ in patch size AND
   objective AND resolution — a cross-model comparison never isolates "patch size". Ablate
   within a family. Render transforms on the mid-gray background (`BG` in the lib), not white
   (white letterbox inflates all similarities and confounds rotation). Compare raw pixels, not
   JPEG round-trips — or report both. (`bb4cb3f`)

6. **The corpus must match its labels.** Filenames must match content — the original `photo-*` set was
   wholesale mislabeled (the "kitten" was a forest path) and every narrative built on the names
   was wrong. If you add images, eyeball them. If you rename, rewrite every reference
   (`test-images/manifest.json`, `FLOOR_REFS` in index.html, `report.html`, `results-*.json`) —
   embeddings don't change on rename, so never re-run batches for one. (`2800941`)

7. **Validate in Node first, then verify the SAME numbers in the browser before pushing.** A
   browser/Node discrepancy is a pipeline artifact until proven otherwise — every "surprising
   model behavior" so far was us, not the model. Batch results (`results-*.json`) are expensive;
   don't regenerate them unless measurement code changed. Single-backend verification is
   incomplete: WASM passing says nothing about WebGPU (q8 CLIP/SigLIP produce garbage there);
   the search page self-calibrates on the user's actual backend for this reason.

8. **Read a model's actual input contract before blaming the runtime.** Gemma 4's ONNX
   "vision_encoder" takes PRE-PATCHIFIED input (`pixel_values [1, seq, 768]` +
   `pixel_position_ids [1, seq, 2]`) — the patchify lives in transformers'
   `Gemma4Processor`, outside the graph. A year of "Gemma 4 doesn't work" was one
   `session.inputNames` inspection away from being solved (plus onnxruntime >= 1.27 for
   GatherBlockQuantized-with-`bits`; and onnxruntime-node 1.27 cannot share a process with
   transformers.js's nested 1.21 — wrong native .so gets dlopened, hence the standalone
   `build-search-index-gemma4.mjs`). Ported preprocessing lives in `lib/gemma4.mjs`, shared
   browser/Node.

## Caching & resilience constraints

Users download hundreds of MB of model weights; they must never download them twice. (`88888f9`)

- **`transformers-cache` belongs to Transformers.js. Never delete it, never duplicate it.** It's
  the only place models land on first visit (before the SW controls the page) and on
  SW-bypassing reloads. The SW's `activate` may delete ONLY its own old `iel-shell-*` caches.
- The SW serves HF model files cache-first via `caches.match()` across ALL caches and lets the
  library do the storing. Only jsdelivr files (the Transformers.js module itself) are stored by
  the SW, in `iel-models`.
- Every page registers `sw.js` AND calls `navigator.storage.persist()` (without it the browser
  may evict gigabytes of cached models under storage pressure).
- Bump `IEL_SW_VERSION` in `sw.js` on every deploy — it drives the update banner.
- Model downloads show per-file progress bars with sizes and speed; failures produce
  network-aware errors ON THE PAGE next to the action (Paul uses this on mobile — there is no
  console). (`e7da581`, `9c66594`, `6a2246b`)

## Design constraints

Every page should look and behave like the same product.

- **Single-file pages, no frameworks, no build step.** Pure-math/shared logic goes in
  `lib/*.mjs` (must run in both browser and Node); pages import it as ES modules.
- **Shared look:** copy the `:root` CSS variable block from an existing page — cream/dark paper
  palette (`--background`, `--bg-secondary`, `--border`, `--muted`), accent `#4b3aff` (dark
  mode `#8ab4f8`), light+dark via `color-scheme` + `prefers-color-scheme`. Georgia serif for
  `h1`/`h2`, system sans body. Data canvases get a 1px `--border` and
  `image-rendering: pixelated` when showing per-dimension pixels.
- **Shared components:** `.links` pill row cross-linking every page (each new page gets added to
  every other page's row + README + docs), `.controls`/`.control-group` with uppercase labels,
  `.status` blocks, `.download-panel` with per-file bars. Reuse them; don't invent parallel
  patterns.
- **Errors and progress appear next to the action that caused them,** terse text, no
  console-only failures.
- **No emoji in new UI elements** — text or inline SVG. (Existing link-row glyphs are
  grandfathered.)
- Layouts must collapse to one column around 700px (Paul drives these pages from his phone).
- Prose: no filler ("honestly", "delve", hype adjectives). The README is a learning log —
  record overturned hypotheses as what they are; a negative result reported straight beats a
  polished wrong claim. (`040866c`, `4e0800f`)

## Workflow

- This is Paul's **live working copy** with an auto-committer and occasional other agents:
  `git pull` before working, push when done, keep commits small and focused.
- `node scripts/test-vector-viz.mjs` (and any other `scripts/test-*.mjs`) must pass before
  pushing lib changes. New lib functions get tests in the same commit.
- README/docs updates ship in the same commit as the feature they describe.
- One-off investigation scripts are kept in `scripts/` (e.g. `probe-florence-rotation.mjs`) —
  they're the reproducible record of why a decision was made.
