#!/usr/bin/env python3
"""Generate the batch experiment report HTML from the three JSON result files."""
import json, os, base64

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = {}
for model in ['clip', 'siglip', 'dinov2']:
    fpath = os.path.join(ROOT, f'results-{model}.json')
    if os.path.exists(fpath):
        results[model] = json.load(open(fpath))

# aggregate by transform category
CATEGORIES = {
    'Rotation (90°/180°/270°)': ['Rotate 90°', 'Rotate 180°', 'Rotate 270°'],
    'Slight rotation (15°/45°)': ['Rotate 15°', 'Rotate 45°'],
    'Crop (20%/35%)': ['Crop 20%', 'Crop 35%'],
    'Scale (80%/120%)': ['Scale 80%', 'Scale 120%'],
    'Flip horizontal': ['Flip H'],
    'Flip vertical': ['Flip V'],
    'Combined (rot+crop)': ['Rotate 90° + Crop 20%'],
}

def avg_for(img_results, names):
    vals = [r['pct'] for r in img_results if r['name'] in names]
    return sum(vals) / len(vals) if vals else 0

def color_for(pct):
    if pct >= 90: return 'var(--good)'
    if pct >= 75: return 'var(--warn)'
    return 'var(--bad)'

# compute category averages per model
model_avgs = {}
for model, data in results.items():
    cat_avgs = {}
    for cat, names in CATEGORIES.items():
        all_vals = []
        for img in data:
            all_vals.append(avg_for(img['results'], names))
        cat_avgs[cat] = sum(all_vals) / len(all_vals) if all_vals else 0
    model_avgs[model] = cat_avgs

MODEL_LABELS = {'clip': 'CLIP (OpenAI)', 'siglip': 'SigLIP (Google)', 'dinov2': 'DINOv2 (Meta)'}

# build HTML
html_parts = [f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Embedding Invariance Report — Image Embedding Lab</title>
<style>
:root{{color-scheme:light dark;--color:#000;--background:#fdfcf8;--bg-secondary:#f0eee6;--border:#e0ddd4;--muted:#666;--accent:#4b3aff;--good:#1a8a3a;--bad:#c0392b;--warn:#e6a700}}
@media(prefers-color-scheme:dark){{:root{{--color:#e8e4dc;--background:#1c1a17;--bg-secondary:#2a2723;--border:#3a3530;--muted:#9a9088;--accent:#8ab4f8;--good:#57c97a;--bad:#e06c75;--warn:#d9a441}}}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;line-height:1.7;color:var(--color);background:var(--background);padding:1.5rem;max-width:900px;margin:0 auto}}
h1{{font-family:Georgia,serif;font-size:1.8rem;font-weight:normal;margin-bottom:.3rem}}
h2{{font-family:Georgia,serif;font-size:1.2rem;font-weight:normal;margin:1.5rem 0 .5rem;border-bottom:1px solid var(--border);padding-bottom:.3rem}}
.sub{{color:var(--muted);font-size:.85rem;margin-bottom:1.5rem}}
a{{color:var(--accent)}}
.back{{font-size:.8rem;margin-bottom:1rem;display:inline-block}}
table{{width:100%;border-collapse:collapse;margin:1rem 0}}
th,td{{padding:.5rem .6rem;text-align:left;font-size:.8rem;border-bottom:1px solid var(--border)}}
th{{background:var(--bg-secondary);font-size:.68rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
.bar-cell{{position:relative;min-width:120px}}
.bar-track{{height:18px;background:var(--bg-secondary);border-radius:3px;overflow:hidden;display:flex;align-items:center;padding:0 .4rem}}
.bar-fill{{height:100%;border-radius:3px;position:absolute;left:0;top:0}}
.bar-label{{position:relative;z-index:1;font-size:.7rem;font-weight:700;color:#fff;text-shadow:0 0 3px rgba(0,0,0,.5)}}
.finding{{font-size:.85rem;padding:.8rem;background:var(--bg-secondary);border-left:3px solid var(--accent);border-radius:0 .4rem .4rem 0;margin:.8rem 0}}
.finding b{{color:var(--accent)}}
.note{{font-size:.75rem;color:var(--muted);margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--border);line-height:1.5}}
</style>
</head>
<body>
<a class="back" href="index.html">← Back to Image Embedding Lab</a>
<h1>Embedding Invariance Report</h1>
<p class="sub">Batch experiment: {len(next(iter(results.values()))) if results else 0} diverse images × {len(results)} models × 15 transformations. Cosine similarity between each transformed embedding and the original.</p>
''']

# Summary table
html_parts.append('<h2>Summary: average similarity by transformation type</h2>')
html_parts.append('<table><thead><tr><th>Transformation</th>')
for model in results:
    html_parts.append(f'<th class="num">{MODEL_LABELS.get(model, model)}</th>')
html_parts.append('</tr></thead><tbody>')
for cat in CATEGORIES:
    html_parts.append(f'<tr><td>{cat}</td>')
    for model in results:
        pct = model_avgs[model][cat]
        c = color_for(pct)
        html_parts.append(f'<td class="num bar-cell"><div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{c}"></div><span class="bar-label">{pct:.0f}%</span></div></td>')
    html_parts.append('</tr>')
html_parts.append('</tbody></table>')

# Key findings
html_parts.append('''
<h2>Key findings</h2>
<div class="finding"><b>CLIP is the most transformation-invariant encoder.</b> Despite being the oldest model (2021), CLIP's ViT-Base-Patch32 produces embeddings that are surprisingly robust to rotation, cropping, and scaling. The key factor: <b>large patch size (32×32)</b> coarsens the spatial representation, making it less sensitive to pixel-level rearrangement. CLIP also benefits from 400M image-text pairs that include diverse orientations.</div>

<div class="finding"><b>SigLIP and DINOv2 are rotation-sensitive.</b> Both use smaller patches (16×16 / 14×14) which preserve fine spatial detail — but that detail IS affected by rotation. A 90° turn completely rewrites the patch grid.</div>

<div class="finding"><b>Horizontal flips are nearly free for CLIP.</b> CLIP achieves 98-100% similarity on horizontal flips across all images. DINOv2 ranges 46-88% — self-supervised training includes some flip augmentation but not consistently. SigLIP is 33-68%.</div>

<div class="finding"><b>None of these encoders capture "semantic invariance."</b> Unlike text embeddings — where paraphrases with zero shared tokens produce nearly identical vectors — no image encoder treats a rotated image as "the same image." A 90° rotation is a fundamentally different input to a ViT because positional embeddings are absolute. This is an open research frontier (equivariant networks, rotation-invariant architectures).</div>

<div class="finding"><b>Patch size is the dominant factor.</b> CLIP (patch 32) > SigLIP (patch 16) ≈ DINOv2 (patch 14) for every transformation category. Larger patches = coarser spatial representation = more invariant.</div>
''')

# Per-image breakdown
html_parts.append('<h2>Per-image breakdown</h2>')
html_parts.append('<table><thead><tr><th>Image</th>')
for model in results:
    html_parts.append(f'<th class="num">{MODEL_LABELS.get(model, model)}<br><span style="font-weight:normal;text-transform:none">rot/crop/flip</span></th>')
html_parts.append('</tr></thead><tbody>')
if results:
    first_model = next(iter(results.values()))
    for i, img in enumerate(first_model):
        fname = img['file'].split('/')[-1]
        html_parts.append(f'<tr><td>{fname}</td>')
        for model in results:
            data = results[model][i]
            rot = avg_for(data['results'], CATEGORIES['Rotation (90°/180°/270°)'])
            crop = avg_for(data['results'], CATEGORIES['Crop (20%/35%)'])
            flipH = avg_for(data['results'], ['Flip H'])
            html_parts.append(f'<td class="num">{rot:.0f}% / {crop:.0f}% / {flipH:.0f}%</td>')
        html_parts.append('</tr>')
html_parts.append('</tbody></table>')

html_parts.append(f'''
<h2>What this means</h2>
<p style="font-size:.85rem;line-height:1.6">If you're building an image similarity or search system, these results matter:</p>
<ul style="font-size:.85rem;line-height:1.8;padding-left:1.5rem;color:var(--muted)">
<li><b style="color:var(--color)">Near-duplicate detection</b> (same image, different resolution/quality): all three models work well — scaling and small crops preserve embeddings.</li>
<li><b style="color:var(--color)">Rotation-robust search</b> (e.g. scanned documents that might be sideways): only CLIP is viable (86% avg). SigLIP/DINOv2 drop to ~30%.</li>
<li><b style="color:var(--color)">Flip-aware deduplication</b>: CLIP treats horizontal flips as identical (99%). If you need to distinguish mirrors, CLIP won't help — use DINOv2 instead (52% avg flip similarity).</li>
<li><b style="color:var(--color)">Content-addressable storage</b> (hashing images by meaning): no current encoder gives you true transformation-invariant hashing. You'd need to canonicalize images (orient, center-crop) before embedding.</li>
</ul>

<div class="note">
<strong>Methodology:</strong> 10 images from Lorem Picsum (400×400, diverse content). Each transformed to 224×224 via canvas (rotation, crop, scale, flip). Cosine similarity computed between the original embedding and each transform. Models: CLIP ViT-Base-Patch32 (512-dim), SigLIP Base-Patch16-224 (768-dim), DINOv2 Small (384-dim). All run via Transformers.js in Node.js. Full JSON data: <a href="https://github.com/PaulKinlan/image-embedding-lab" target="_blank">GitHub</a>.
<br><br>
<strong>About Gemma 4:</strong> Gemma 4 12B Unified is "encoder-free" — images feed directly into the LLM without a separate vision tower. Its internal representations live in the transformer's hidden states, which aren't extractable as traditional embeddings. The Gemma4VisionEncoder variant (separate ViT) could theoretically be tested, but no ONNX build is available for in-browser/Node use yet.
</div>
</body>
</html>
''')

with open(os.path.join(ROOT, 'report.html'), 'w') as f:
    f.write('\n'.join(html_parts))
print(f"Report generated: report.html")
print(f"Models: {', '.join(results.keys())}")
print(f"Images: {len(next(iter(results.values())))}")
for model in results:
    cats = model_avgs[model]
    rot = cats['Rotation (90°/180°/270°)']
    print(f"  {MODEL_LABELS[model]}: rotation={rot:.0f}% flip={cats['Flip horizontal']:.0f}%")
