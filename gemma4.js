// Gemma 4 Vision Encoder — loaded directly via ONNX Runtime Web
// The vision_encoder_q4.onnx is a standalone submodel of Gemma 4 E2B.
// It outputs 280 soft tokens (768-dim each) that we mean-pool into an embedding.

const GEMMA4_VISION_URL = 'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/vision_encoder_q4.onnx';
const GEMMA4_VISION_DATA_URL = 'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/vision_encoder_q4.onnx_data';
const INPUT_SIZE = 224; // standard ViT resolution

let gemma4Session = null;
let gemma4InputName = null;
let gemma4OutputName = null;

async function loadGemma4(progressCb) {
  if (gemma4Session) return gemma4Session;

  // Load ORT from CDN
  if (typeof ort === 'undefined') {
    await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js');
  }

  // Fetch model files with progress
  progressCb?.({ status: 'download', file: 'vision_encoder_q4.onnx', loaded: 0, total: 196000 });
  const modelBuf = await fetch(GEMMA4_VISION_URL).then(r => r.arrayBuffer());

  progressCb?.({ status: 'download', file: 'vision_encoder_q4.onnx_data', loaded: 0, total: 107000000 });
  const dataBuf = await fetchWithProgress(GEMMA4_VISION_DATA_URL, (loaded, total) => {
    progressCb?.({ status: 'progress', file: 'vision_encoder_q4.onnx_data', loaded, total });
  });

  progressCb?.({ status: 'done', file: 'vision_encoder_q4.onnx' });
  progressCb?.({ status: 'done', file: 'vision_encoder_q4.onnx_data' });

  // Create session with external data
  gemma4Session = await ort.InferenceSession.create(modelBuf, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    externalData: [{ path: 'vision_encoder_q4.onnx_data', data: dataBuf }],
  });

  // Find input/output names
  gemma4InputName = gemma4Session.inputNames[0]; // typically 'pixel_values'
  gemma4OutputName = gemma4Session.outputNames[0]; // typically 'last_hidden_state' or similar

  return gemma4Session;
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const total = parseInt(resp.headers.get('content-length') || '0');
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
  // Resize to 224x224 (already done by our renderTransform which uses 224x224 canvases)
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = imageData.data; // RGBA

  // Convert to NCHW float32 tensor: [1, 3, 224, 224]
  // SigLIP/Gemma normalization: mean=0.5, std=0.5 for each channel
  const tensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const planeSize = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < planeSize; i++) {
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    // Normalize: (x - 0.5) / 0.5
    tensor[i] = (r - 0.5) / 0.5;                  // R channel plane
    tensor[planeSize + i] = (g - 0.5) / 0.5;       // G channel plane
    tensor[planeSize * 2 + i] = (b - 0.5) / 0.5;   // B channel plane
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

  // Output shape is [1, num_tokens, 768] — mean pool across tokens
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

// Export for use in index.html
window.__gemma4 = { loadGemma4, embedGemma4, isLoaded: () => !!gemma4Session };
