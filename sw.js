// App-shell service worker. Model weights are NOT handled here: the inference worker
// already persists them in its own Cache API bucket ("nemotron-asr-int4-v1").
const SHELL_CACHE = "nemotron-asr-shell-v1";
const ORT_VER = "1.26.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;

const SHELL = [
  "/",
  "/index.html",
  "/manifest.webmanifest",
  "/src/styles.css",
  "/src/app.js",
  "/src/worker.js",
  "/src/shared.js",
  "/src/mic-processor.js",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  "/icons/icon-192-maskable.png",
  "/icons/icon-512-maskable.png",
  "/icons/apple-touch-icon.png",
  ORT_BASE + "ort.webgpu.mjs",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((c) => c.addAll(SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k.startsWith("nemotron-asr-shell-") && k !== SHELL_CACHE)
            .map((k) => caches.delete(k)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const sameOrigin = url.origin === self.location.origin;
  // onnxruntime lazily imports wasm/mjs chunks from the CDN at runtime; cache them too.
  const isOrt = req.url.startsWith(ORT_BASE);
  if (!sameOrigin && !isOrt) return;

  e.respondWith(
    caches.open(SHELL_CACHE).then(async (cache) => {
      const cached = await cache.match(req, { ignoreSearch: sameOrigin });
      const network = fetch(req)
        .then((resp) => {
          if (resp.ok && (resp.type === "basic" || resp.type === "cors")) {
            cache.put(req, resp.clone());
          }
          return resp;
        })
        .catch(() => cached);
      // Stale-while-revalidate: serve cached shell instantly, refresh in the background.
      return cached || network;
    }),
  );
});
