// Experiment 7 — TEXT↔IMAGE ALIGNMENT UNDER TRANSFORMS (CLIP). Vector movement is one thing;
// SEMANTIC survival is another. For each transform, does the image still retrieve its own
// caption (top-1 among all 15 captions, image↔text cosine)? A transform can move the vector a
// long way while leaving it closest to the right caption — that's semantic robustness.
import fs from "node:fs";
import { CORPUS, embed, cosineSim, fmt } from "./lib-harness.mjs";

const CAPTIONS = {
  "photo-cafe.jpg": "a cafe interior with coffee cups on a wooden table",
  "photo-phone-photographer.jpg": "hands taking a photo of a city with a smartphone",
  "photo-road-tunnel.jpg": "an aerial view of a road tunnel surrounded by trees",
  "photo-floating-market.jpg": "a floating market with boats full of food",
  "photo-forest-path.jpg": "a path through a forest",
  "photo-sea-cliff.jpg": "a rocky sea cliff on a black sand beach",
  "photo-open-book.jpg": "an open book on a table",
  "photo-mountain-lake.jpg": "a wooden jetty on a mountain lake",
  "photo-skateboard.jpg": "a skateboard leaning against a wall",
  "photo-ocean-waves.jpg": "waves crashing on rocks",
  "text-readme.png": "a screenshot of a text document",
  "webpage-example.png": "a screenshot of a simple webpage",
  "webpage-hn.png": "a screenshot of a news website with a list of links",
  "webpage-sotw.png": "a screenshot of a colorful webpage",
  "webpage-wikipedia.png": "a screenshot of a wikipedia article",
};

const CONDITIONS = [
  { name: "identity", p: {} },
  { name: "rotate 90°", p: { rotation: 90 } },
  { name: "rotate 180°", p: { rotation: 180 } },
  { name: "rotate 45°", p: { rotation: 45 } },
  { name: "shuffle 4×4", p: { shuffle: 4 } },
  { name: "flip H", p: { flip: "horizontal" } },
  { name: "blur", p: { blur: 3 } },
];

const T = await import("@huggingface/transformers");
const MODEL_ID = "Xenova/clip-vit-base-patch32";
const tok = await T.AutoTokenizer.from_pretrained(MODEL_ID);
const tm = await T.CLIPTextModelWithProjection.from_pretrained(MODEL_ID);
const files = CORPUS.filter((f) => CAPTIONS[f]);
const inputs = tok(files.map((f) => "a photo of " + CAPTIONS[f]), { padding: true, truncation: true });
const { text_embeds } = await tm(inputs);
const D = text_embeds.dims[1];
const textVecs = files.map((_, i) => Array.from(text_embeds.data.slice(i * D, (i + 1) * D)));

const results = {};
for (const cond of CONDITIONS) {
  let top1 = 0;
  let margins = 0;
  for (let i = 0; i < files.length; i++) {
    const v = (await embed("clip", files[i], cond.p)).pooled;
    const scores = textVecs.map((t) => cosineSim(v, t));
    const own = scores[i];
    const bestOther = Math.max(...scores.filter((_, j) => j !== i));
    if (own > bestOther) top1++;
    margins += own - bestOther;
  }
  results[cond.name] = { top1Rate: top1 / files.length, meanMargin: margins / files.length };
  console.log(`${cond.name.padEnd(12)} own-caption top-1 ${top1}/${files.length}  mean margin ${fmt(margins / files.length)}`);
}
fs.writeFileSync("results-text-alignment.json", JSON.stringify({ model: MODEL_ID, captions: CAPTIONS, results }, null, 2));
console.log("\nwritten results-text-alignment.json");
