// Service worker — resilient caching for very large model blobs + network-first app shell.
// Version: bump IEL_SW_VERSION on every deploy to trigger update detection.
//
// CACHE OWNERSHIP RULES (learned the hard way — see AGENTS.md):
// - Transformers.js keeps its OWN Cache Storage cache ('transformers-cache') and writes every
//   model file it downloads there. That cache also covers loads the SW never sees (first visit
//   before the SW controls the page, hard reloads that bypass the SW). NEVER delete or
//   duplicate it: activate() must only clean up our own old shell caches, and the fetch
//   handler serves model files via caches.match() across ALL caches instead of re-storing
//   another copy (double-storing ~GB models is what pushed the origin into quota eviction).
// - Only jsdelivr files (the Transformers.js module itself, ~1MB) are stored by us, in
//   MODEL_CACHE — the library doesn't cache its own JS.
const IEL_SW_VERSION = '2026-07-15-v2';
const MODEL_CACHE = `iel-models`;
const SHELL_CACHE = `iel-shell-${IEL_SW_VERSION}`;
const HF_ORIGINS = ['huggingface.co', 'cdn-lfs.huggingface.co'];
const CDN_ORIGINS = ['cdn.jsdelivr.net'];

self.addEventListener('install', (e) => {
  // skipWaiting so the new SW activates immediately
  self.skipWaiting();
  e.waitUntil(caches.open(SHELL_CACHE).then(() => {}).catch(() => {}));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          // Delete ONLY our own outdated shell caches. Anything else (transformers-cache,
          // iel-models, future library caches) is load-bearing model storage.
          .filter((k) => k.startsWith('iel-shell-') && k !== SHELL_CACHE)
          .map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
  // One-time migration: an earlier SW double-stored HF model files in MODEL_CACHE (the library
  // already keeps them in transformers-cache) — roughly doubling the origin's storage and
  // courting quota eviction. Move any HF entries the library cache is missing, then drop the
  // duplicates.
  e.waitUntil((async () => {
    try {
      const modelCache = await caches.open(MODEL_CACHE);
      const libCache = await caches.open('transformers-cache');
      for (const req of await modelCache.keys()) {
        if (!matchesOrigin(new URL(req.url).hostname, HF_ORIGINS)) continue;
        if (!(await libCache.match(req))) {
          const res = await modelCache.match(req);
          if (res) await libCache.put(req, res);
        }
        await modelCache.delete(req);
      }
    } catch (err) { /* best-effort; never block activation */ }
  })());
});

function matchesOrigin(hostname, origins) {
  return origins.some((o) => hostname === o || hostname.endsWith('.' + o));
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Model files from Hugging Face: serve from ANY cache (Transformers.js writes them to its
  // own 'transformers-cache'); on miss, go to network and let the library do the storing.
  if (matchesOrigin(url.hostname, HF_ORIGINS)) {
    e.respondWith(
      caches.match(req).then((cached) => cached || fetch(req))
    );
    return;
  }

  // CDN files (the Transformers.js module): cache-first, stored by us.
  if (matchesOrigin(url.hostname, CDN_ORIGINS)) {
    e.respondWith(
      caches.match(req).then((cached) => {
        if (cached) return cached;
        return caches.open(MODEL_CACHE).then((cache) =>
          fetch(req).then((res) => {
            if (res && res.status === 200) {
              const copy = res.clone();
              e.waitUntil(cache.put(req, copy).catch(() => {}));
            }
            return res;
          })
        );
      })
    );
    return;
  }

  // App shell (HTML, JS, CSS): network-first so new versions appear immediately
  if (url.origin === self.location.origin) {
    e.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          e.waitUntil(caches.open(SHELL_CACHE).then((c) => c.put(req, copy)).catch(() => {}));
          return res;
        })
        .catch(() => caches.open(SHELL_CACHE).then((c) => c.match(req)).then((r) => r || Response.error()))
    );
  }
});
