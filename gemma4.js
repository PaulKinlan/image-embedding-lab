// Gemma 4 Vision Encoder — WORKING integration (2026-07-15).
//
// Why the first attempt failed, for the record: the ONNX "vision_encoder" does not take an
// image. Its inputs are pixel_values [1, seq, 768] (flattened 16×16×3 patches, channel-last,
// rescaled to [0,1]) and pixel_position_ids [1, seq, 2] ((col,row) per patch) — the patchify
// step lives in transformers' Gemma4Processor, outside the graph. Feeding raw CHW pixels could
// never work. Additionally the q4 graph uses GatherBlockQuantized with a `bits` attribute,
// which needs onnxruntime >= 1.27 (the old pin was 1.20.1). The preprocessing is ported in
// lib/gemma4.mjs and shared with the Node index builder, so browser and index stay aligned.
//
// ort.min.js is a UMD/global bundle: it MUST be loaded via a <script> tag (import() runs it as
// an ES module and never defines the global).

const GEMMA4_BASE = 'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx';
const ORT_VERSION = '1.27.0';

let gemma4Session = null;
let gemma4Lib = null;

async function loadGemma4(progressCb) {
  if (gemma4Session) return gemma4Session;

  if (typeof ort === 'undefined') {
    await new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.min.js`;
      s.onload = resolve;
      s.onerror = () => reject(new Error('Failed to load ONNX Runtime Web from CDN.'));
      document.head.appendChild(s);
    });
  }
  if (typeof ort === 'undefined') throw new Error('ONNX Runtime (ort) global not available after load.');
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

  gemma4Lib = await import('./lib/gemma4.mjs');

  progressCb?.({ status: 'download', file: 'vision_encoder_q4.onnx', loaded: 0, total: 200072 });
  const modelResp = await fetch(`${GEMMA4_BASE}/vision_encoder_q4.onnx`);
  if (!modelResp.ok) throw new Error(`Failed to fetch model: HTTP ${modelResp.status}`);
  const modelBuf = await modelResp.arrayBuffer();
  progressCb?.({ status: 'done', file: 'vision_encoder_q4.onnx' });

  progressCb?.({ status: 'download', file: 'vision_encoder_q4.onnx_data', loaded: 0, total: 112105664 });
  const dataBuf = await fetchWithProgress(`${GEMMA4_BASE}/vision_encoder_q4.onnx_data`, (loaded, total) => {
    progressCb?.({ status: 'progress', file: 'vision_encoder_q4.onnx_data', loaded, total });
  });
  progressCb?.({ status: 'done', file: 'vision_encoder_q4.onnx_data' });

  try {
    gemma4Session = await ort.InferenceSession.create(modelBuf, {
      executionProviders: ['wasm'],
      externalData: [{ path: 'vision_encoder_q4.onnx_data', data: new Uint8Array(dataBuf) }],
    });
  } catch (e) {
    const code = (e && e.message) ? e.message : String(e);
    throw new Error(`Gemma 4 failed to initialize in ONNX Runtime ${ORT_VERSION} (${code}).`);
  }
  console.log('[gemma4] session ready. inputs:', gemma4Session.inputNames, 'outputs:', gemma4Session.outputNames);
  return gemma4Session;
}

// Embed a canvas: pixels at the canvas's own resolution; the shared lib resizes to the model's
// patch budget (dims divisible by 48), patchifies, and mean-pools the 1536-d soft tokens —
// exactly like the offline index builder.
async function embedGemma4(canvas) {
  await loadGemma4();
  const ctx = canvas.getContext('2d');
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return gemma4Lib.gemma4Embed(gemma4Session, ort, rgba);
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const total = parseInt(resp.headers.get('content-length') || '112105664');
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

window.__gemma4 = { loadGemma4, embedGemma4, isLoaded: () => !!gemma4Session };
