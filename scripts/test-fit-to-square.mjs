// Tests for lib/experiment.mjs fitToSquare — the shared pure-JS resampler that keeps the
// search index and the browser query on identical pixels. Run: node scripts/test-fit-to-square.mjs
import { fitToSquare } from "../lib/experiment.mjs";

let failures = 0;
const check = (name, cond) => { if (!cond) { failures++; console.error("FAIL:", name); } else console.log("ok:", name); };

// solid color survives, letterbox is mid-gray, output dims correct
const solid = { width: 100, height: 50, data: new Uint8ClampedArray(100 * 50 * 4).fill(200) };
for (let i = 3; i < solid.data.length; i += 4) solid.data[i] = 255;
const out = fitToSquare(solid, 224);
check("output is size×size", out.width === 224 && out.height === 224 && out.data.length === 224 * 224 * 4);
check("center color preserved", out.data[(112 * 224 + 112) * 4] === 200);
check("letterbox is mid-gray", out.data[(2 * 224 + 112) * 4] === 127);
check("alpha opaque", out.data.every((v, i) => i % 4 !== 3 || v === 255));

// area-average preserves the mean (energy conservation, square source = no letterbox)
const rnd = { width: 384, height: 384, data: new Uint8ClampedArray(384 * 384 * 4) };
let s = 7;
for (let i = 0; i < rnd.data.length; i++) { s = (s * 16807) % 2147483647; rnd.data[i] = i % 4 === 3 ? 255 : s % 256; }
const small = fitToSquare(rnd, 224);
const mean = (d, ch) => { let t = 0, n = 0; for (let i = ch; i < d.length; i += 4) { t += d[i]; n++; } return t / n; };
check("mean preserved on downscale", Math.abs(mean(rnd.data, 0) - mean(small.data, 0)) < 0.5);

// deterministic
const a = fitToSquare(rnd, 64), b = fitToSquare(rnd, 64);
check("deterministic", a.data.every((v, i) => v === b.data[i]));

// upscale path (bilinear) stays in range and fills the square
const tiny = { width: 8, height: 8, data: new Uint8ClampedArray(8 * 8 * 4).fill(100) };
for (let i = 3; i < tiny.data.length; i += 4) tiny.data[i] = 255;
const up = fitToSquare(tiny, 64);
check("upscale fills square with source color", up.data[(32 * 64 + 32) * 4] === 100);

console.log(failures ? `\n${failures} FAILURES` : "\nAll tests passed");
process.exit(failures ? 1 : 0);
