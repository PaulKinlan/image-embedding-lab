// Service worker — caches large model blobs (HuggingFace CDN) and the page shell.
// Stale-while-revalidate: serve from cache immediately, update in background.
// Version-keyed: bump MODEL_CACHE_VERSION to force re-download of model files.
const MODEL_CACHE_VERSION = 'v1';
const MODEL_CACHE = `iel-models-${MODEL_CACHE_VERSION}`;
const SHELL_CACHE = `iel-shell-${MODEL_CACHE_VERSION}`;
const MODEL_ORIGINS = ['huggingface.co', 'cdn-lfs.huggingface.co', 'cdn.jsdelivr.net'];

self.addEventListener('install', (e) => {
  self.skipWaiting();
  e.waitUntil(caches.open(SHELL_CACHE).then((c) => c.addAll(['/', '/index.html', '/sw.js'])).catch(() => {}));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== MODEL_CACHE && k !== SHELL_CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Cache model files from HuggingFace + CDN (large blobs, download once)
  const isModelFile = MODEL_ORIGINS.some((o) => url.hostname === o || url.hostname.endsWith('.' + o));
  if (isModelFile) {
    e.respondWith(
      caches.open(MODEL_CACHE).then(async (cache) => {
        const cached = await cache.match(req);
        const network = fetch(req).then((res) => {
          if (res && res.status === 200) cache.put(req, res.clone());
          return res;
        }).catch(() => cached || Response.error());
        // Return cached immediately if available, otherwise wait for network
        return cached || network;
      })
    );
    return;
  }

  // Same-origin: stale-while-revalidate for the app shell
  if (url.origin === self.location.origin) {
    e.respondWith(
      caches.open(SHELL_CACHE).then(async (cache) => {
        const cached = await cache.match(req);
        const network = fetch(req).then((res) => {
          if (res && res.status === 200) cache.put(req, res.clone());
          return res;
        }).catch(() => cached);
        return cached || network;
      })
    );
  }
});
