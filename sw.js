// Service worker — caches large model blobs, network-first for app shell.
// Version: bump IEL_SW_VERSION on every deploy to trigger update detection.
const IEL_SW_VERSION = '2026-07-14-v1';
const MODEL_CACHE = `iel-models`;
const SHELL_CACHE = `iel-shell-${IEL_SW_VERSION}`;
const MODEL_ORIGINS = ['huggingface.co', 'cdn-lfs.huggingface.co', 'cdn.jsdelivr.net'];

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
          .filter((k) => !k.startsWith('iel-models') && k !== SHELL_CACHE)
          .map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Model files: cache-first (large blobs, download once, serve instantly)
  const isModelFile = MODEL_ORIGINS.some((o) => url.hostname === o || url.hostname.endsWith('.' + o));
  if (isModelFile) {
    e.respondWith(
      caches.open(MODEL_CACHE).then(async (cache) => {
        const cached = await cache.match(req);
        const network = fetch(req).then((res) => {
          if (res && res.status === 200) cache.put(req, res.clone());
          return res;
        }).catch(() => cached || Response.error());
        return cached || network;
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
          caches.open(SHELL_CACHE).then((c) => c.put(req, copy)).catch(() => {});
          return res;
        })
        .catch(() => caches.open(SHELL_CACHE).then((c) => c.match(req)).then((r) => r || Response.error()))
    );
  }
});
