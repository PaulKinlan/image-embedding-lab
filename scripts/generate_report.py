#!/usr/bin/env python3
"""Generate the batch experiment report HTML from the three JSON result files.

Consumes the current schema (see cli.mjs): each results-<model>.json has
{model, images, baseline:{raw,jpeg}, summary:[{name, raw:{mean,std,n}, jpeg:{...}}],
perImage:[{file, results:[{name, raw, jpeg}]}]}. Embeddings are mean-pooled + normalized.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = {}
for model in ['clip', 'siglip', 'dinov2']:
    fpath = os.path.join(ROOT, f'results-{model}.json')
    if os.path.exists(fpath):
        results[model] = json.load(open(fpath))

CATEGORIES = {
    'Rotation (90/180/270)': ['Rotate 90°', 'Rotate 180°', 'Rotate 270°'],
    'Slight rotation (15/45)': ['Rotate 15°', 'Rotate 45°'],
    'Crop (20/35%)': ['Crop 20%', 'Crop 35%'],
    'Scale (80/120%)': ['Scale 80%', 'Scale 120%'],
    'Flip horizontal': ['Flip H'],
    'Flip vertical': ['Flip V'],
    'Grayscale': ['Grayscale'],
    'Brightness (±30%)': ['Brighten 1.3×', 'Darken 0.7×'],
    'Blur': ['Blur'],
    'Occlude 25%': ['Occlude 25%'],
}

def summ_mean(data, names):
    vals = [t['raw']['mean'] for t in data['summary']
            if t['name'] in names and t['raw']['mean'] is not None]
    return sum(vals) / len(vals) * 100 if vals else 0

def img_avg(img_results, names, key='raw'):
    vals = [r[key] for r in img_results if r['name'] in names and r[key] is not None]
    return sum(vals) / len(vals) * 100 if vals else 0

def floor_of(data):
    return (data['baseline']['raw']['mean'] or 0) * 100

def color_for(pct):
    if pct >= 90: return 'var(--good)'
    if pct >= 70: return 'var(--warn)'
    return 'var(--bad)'

model_avgs = {m: {cat: summ_mean(d, names) for cat, names in CATEGORIES.items()}
              for m, d in results.items()}
floors = {m: floor_of(d) for m, d in results.items()}

MODEL_LABELS = {'clip': 'CLIP (OpenAI)', 'siglip': 'SigLIP (Google)', 'dinov2': 'DINOv2 (Meta)'}
n_images = len(next(iter(results.values()))['perImage']) if results else 0

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
.floor-row td{{font-weight:700;border-top:2px solid var(--border)}}
.finding{{font-size:.85rem;padding:.8rem;background:var(--bg-secondary);border-left:3px solid var(--accent);border-radius:0 .4rem .4rem 0;margin:.8rem 0}}
.finding b{{color:var(--accent)}}
.note{{font-size:.75rem;color:var(--muted);margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--border);line-height:1.5}}
</style>
</head>
<body>
<a class="back" href="index.html">← Back to Image Embedding Lab</a>
<h1>Embedding Invariance Report</h1>
<p class="sub">{n_images} diverse images × {len(results)} encoders × {len(CATEGORIES)} transform groups. Cosine similarity between each transformed embedding and the original — <b>mean-pooled + L2-normalized</b>, so the models are compared like-for-like. Each cell is the raw-pixel mean; the different-image floor tells you how low "low" is.</p>
''']

# Summary table
html_parts.append('<h2>Summary: mean similarity by transform (raw pixels)</h2>')
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
# baseline floor row
html_parts.append('<tr class="floor-row"><td>Different-image floor</td>')
for model in results:
    html_parts.append(f'<td class="num">{floors[model]:.0f}%</td>')
html_parts.append('</tr>')
html_parts.append('</tbody></table>')

# Key findings (corrected)
def norm(model, cat):
    f = floors[model]
    return (model_avgs[model][cat] - f) / (100 - f) * 100 if (100 - f) else 0

rot_norm = {m: norm(m, 'Rotation (90/180/270)') for m in results}
html_parts.append(f'''
<h2>Key findings</h2>
<div class="finding"><b>Patch size does <i>not</i> rank the encoders.</b> With embeddings pooled, SigLIP (patch 16) matches CLIP (patch 32) on rotation, and DINOv2 (patch 14) is close behind. An earlier version of this report compared CLIP's pooled vector against SigLIP/DINOv2's raw <i>unpooled patch grids</i> — rotating an image scrambles a patch grid's token order, which manufactured a fake "small patch = rotation-sensitive" effect. Pooling removes it.</div>

<div class="finding"><b>The baseline changes the ranking.</b> The different-image floor differs a lot — CLIP {floors.get('clip',0):.0f}%, SigLIP {floors.get('siglip',0):.0f}%, DINOv2 {floors.get('dinov2',0):.0f}%. So CLIP's high absolute scores are partly a compressed embedding space (two unrelated images already sit at {floors.get('clip',0):.0f}%). Normalized to each model's own floor, rotation robustness is CLIP {rot_norm.get('clip',0):.0f}%, SigLIP {rot_norm.get('siglip',0):.0f}%, DINOv2 {rot_norm.get('dinov2',0):.0f}% — comparable. "CLIP is the most invariant" doesn't survive the baseline.</div>

<div class="finding"><b>JPEG q0.9 is negligible.</b> Every transform is embedded twice (raw pixels and a JPEG round-trip); the mean difference is ~1 point. Compression is not what these numbers measure.</div>

<div class="finding"><b>Rotation is the hardest geometric transform — but not catastrophic.</b> 72–85% raw, comfortably above every floor. Photometric changes (grayscale, brightness, blur, occlusion) are nearly free (high 80s–90s): these encoders care about content layout far more than exact pixels.</div>

<div class="finding"><b>Still true:</b> no encoder is fully transformation-invariant — a rotated image is not treated as identical. ViT positional embeddings are absolute, so a turn is a different input. That framing was right in the earlier write-up; only the magnitudes were wrong.</div>
''')

# Per-image breakdown
html_parts.append('<h2>Per-image breakdown (rotation / crop / flip-H, raw)</h2>')
html_parts.append('<table><thead><tr><th>Image</th>')
for model in results:
    html_parts.append(f'<th class="num">{MODEL_LABELS.get(model, model)}</th>')
html_parts.append('</tr></thead><tbody>')
if results:
    first = next(iter(results.values()))['perImage']
    for i, img in enumerate(first):
        fname = img['file'].split('/')[-1]
        html_parts.append(f'<tr><td>{fname}</td>')
        for model in results:
            r = results[model]['perImage'][i]['results']
            rot = img_avg(r, CATEGORIES['Rotation (90/180/270)'])
            crop = img_avg(r, CATEGORIES['Crop (20/35%)'])
            flipH = img_avg(r, ['Flip H'])
            html_parts.append(f'<td class="num">{rot:.0f}% / {crop:.0f}% / {flipH:.0f}%</td>')
        html_parts.append('</tr>')
html_parts.append('</tbody></table>')

html_parts.append('''
<div class="note">
<strong>Embedding method:</strong> each image is rendered to a square on a mid-gray background, passed to the vision encoder, and its output is reduced to ONE vector, then L2-normalized — that vector is what the cosine compares. The pooling step matters: <code>image-feature-extraction</code> returns a pooled <code>[1,512]</code> for CLIP but raw patch-token grids for SigLIP <code>[1,196,768]</code> and DINOv2 <code>[1,257,384]</code>; mean-pooling every model's tokens to one vector is what makes them comparable (skipping it compares flattened patch grids, where rotation just reshuffles token order — the artifact that produced the earlier "patch size" result). Every transform is embedded at raw pixels and three JPEG qualities (q0.9 / q0.5 / q0.2); tiling optionally splits the image into crops, embeds each, and pools them. The floor is the mean cosine between distinct images' originals.
<br><br>
<strong>Same method on a VLM:</strong> a vision-language model's encoder builds the same token grid, so you can tap it and pool it identically — Florence-2's <code>encode_image</code> gives <code>[1,577,768]</code> → mean-pool → a 768-d image embedding. An embedding is just one pooled hidden state; the alternative is running that hidden state through the output layer to write text. Try it in the <a href="vlm.html">VLM playground</a>.
<br><br>
Reproduce: <code>node cli.mjs test-images/* --model clip|siglip|dinov2|siglip384|clip-l --tiles N --json</code>. Data + code: <a href="https://github.com/PaulKinlan/image-embedding-lab" target="_blank">GitHub</a>.
</div>
</body>
</html>
''')

with open(os.path.join(ROOT, 'report.html'), 'w') as f:
    f.write('\n'.join(html_parts))
print("Report generated: report.html")
for model in results:
    print(f"  {MODEL_LABELS[model]}: rotation={model_avgs[model]['Rotation (90/180/270)']:.0f}% "
          f"floor={floors[model]:.0f}%")
