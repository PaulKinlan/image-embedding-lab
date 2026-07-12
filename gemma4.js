// Gemma 4 Vision Encoder — loaded via ONNX Runtime Web
// The vision_encoder_q4.onnx is a standalone submodel of Gemma 4 E2B.
// Outputs 280 soft tokens (768-dim) that we mean-pool into an embedding.

const GEMMA4_BASE = 'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx';
const GEMMA4_MODEL_URL = `${GEMMA4_BASE}/vision_encoder_q4.onnx`;
const GEMMA4_DATA_URL = `${GEMMA4_BASE}/vision_encoder_q4.onnx_data`;
const INPUT_SIZE = 224;

let gemma4Session = null;
let gemma4InputName = null;
let gemma4OutputName = null;

async function loadGemma4(progressCb) {
  if (gemma4Session) return gemma4Session;

  // Load ONNX Runtime Web. ort.min.js is a UMD/global bundle, so it MUST be loaded via a
  // <script> tag (which sets window.ort). A dynamic import() runs it as an ES module and never
  // defines the global — that's the "ort is not defined" error.
  if (typeof ort === 'undefined') {
    await new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';
      s.onload = resolve;
      s.onerror = () => reject(new Error('Failed to load ONNX Runtime Web from CDN.'));
      document.head.appendChild(s);
    });
  }
  if (typeof ort === 'undefined') throw new Error('ONNX Runtime (ort) global not available after load.');
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

  // Fetch model graph (small, ~196KB)
  progressCb?.({ status: 'download', file: 'vision_encoder_q4.onnx', loaded: 0, total: 196608 });
  const modelResp = await fetch(GEMMA4_MODEL_URL);
  if (!modelResp.ok) throw new Error(`Failed to fetch model: HTTP ${modelResp.status}`);
  const modelBuf = await modelResp.arrayBuffer();
  progressCb?.({ status: 'done', file: 'vision_encoder_q4.onnx' });

  // Fetch model weights (large, ~107MB) with progress
  progressCb?.({ status: 'download', file: 'vision_encoder_q4.onnx_data', loaded: 0, total: 107000000 });
  const dataBuf = await fetchWithProgress(GEMMA4_DATA_URL, (loaded, total) => {
    progressCb?.({ status: 'progress', file: 'vision_encoder_q4.onnx_data', loaded, total });
  });
  progressCb?.({ status: 'done', file: 'vision_encoder_q4.onnx_data' });

  // Create session — pass external data as Uint8Array
  // The model graph references config.json internally; intercept that fetch
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async function (url, opts) {
    // ORT tries to fetch config.json from a wrong path — return empty JSON
    if (typeof url === 'string' && url.includes('config.json')) {
      return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    return originalFetch.call(this, url, opts);
  };

  try {
    gemma4Session = await ort.InferenceSession.create(modelBuf, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      externalData: [{
        path: 'vision_encoder_q4.onnx_data',
        data: new Uint8Array(dataBuf),
      }],
    });
  } finally {
    globalThis.fetch = originalFetch;
  }

  gemma4InputName = gemma4Session.inputNames[0];
  gemma4OutputName = gemma4Session.outputNames[0];
  return gemma4Session;
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const total = parseInt(resp.headers.get('content-length') || '110000000');
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (onProgress) onProgress(loaded, total);
  }
  const result = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result.buffer;
}

function preprocessForGemma4(canvas) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = imageData.data;

  const tensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const planeSize = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < planeSize; i++) {
    tensor[i] = (pixels[i * 4] / 255 - 0.5) / 0.5;
    tensor[planeSize + i] = (pixels[i * 4 + 1] / 255 - 0.5) / 0.5;
    tensor[planeSize * 2 + i] = (pixels[i * 4 + 2] / 255 - 0.5) / 0.5;
  }
  return tensor;
}

async function embedGemma4(canvas) {
  await loadGemma4();
  const tensorData = preprocessForGemma4(canvas);
  const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const feeds = {};
  feeds[gemma4InputName] = inputTensor;
  const results = await gemma4Session.run(feeds);
  const output = results[gemma4OutputName];

  const dims = output.dims;
  const data = output.data;
  const numTokens = dims[dims.length - 2] || 280;
  const embedDim = dims[dims.length - 1] || 768;

  const pooled = new Float64Array(embedDim);
  for (let t = 0; t < numTokens; t++) {
    for (let d = 0; d < embedDim; d++) {
      pooled[d] += data[t * embedDim + d];
    }
  }
  for (let d = 0; d < embedDim; d++) pooled[d] /= numTokens;
  return pooled;
}

window.__gemma4 = { loadGemma4, embedGemma4, isLoaded: () => !!gemma4Session };
