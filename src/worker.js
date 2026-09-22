import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.webgpu.mjs";
import {
  CONFIG,
  buildMelFB,
  buildWindow,
  computeMelOffline,
  StreamingMel,
  detok,
} from "./shared.js";

const C = CONFIG;
const ORT_VER = "1.26.0";

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const CACHE_NAME = "nemotron-asr-int4-v1";
const IS_MOBILE = /Mobi|Android|iPhone|iPad|iPod/i.test(
  (self.navigator && navigator.userAgent) || "",
);

const post = (m, t) => self.postMessage(m, t || []);

let ENC, DEC, JOINT, VOCAB, MELFB, WINDOW;
let ready = false;
let initInFlight = null;
let ENC_EP = "webgpu";

async function openCache() {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null;
  }
}

function progressStream(body, label, total) {
  let loaded = 0;
  return body.pipeThrough(
    new TransformStream({
      transform(chunk, ctrl) {
        loaded += chunk.byteLength;
        post({ type: "progress", label, loaded, total });
        ctrl.enqueue(chunk);
      },
    }),
  );
}

async function fetchBlobCached(url, label) {
  const cache = await openCache();
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      post({ type: "progress", label, loaded: 1, total: 1, cached: true });
      return await hit.blob();
    }
  }
  // huggingface.co returns 404 (without CORS headers) when the Referer is a *.workers.dev origin
  const fetchOpts = { referrerPolicy: "no-referrer" };
  let resp = await fetch(url, fetchOpts);
  if (!resp.ok || !resp.body)
    throw new Error(`HTTP ${resp.status || "?"} fetching ${label}`);
  const total = Number(resp.headers.get("content-length")) || 0;

  if (cache) {
    try {
      await cache.put(
        url,
        new Response(progressStream(resp.body, label, total), {
          headers: {
            "Content-Type": "application/octet-stream",
            "Content-Length": String(total),
          },
        }),
      );
      const hit = await cache.match(url);
      if (hit) return await hit.blob();
    } catch (e) {
      post({
        type: "status",
        detail: `cache unavailable (${(e && e.name) || e}); loading without cache`,
      });
      resp = await fetch(url, fetchOpts);
      if (!resp.ok || !resp.body)
        throw new Error(`HTTP ${resp.status || "?"} re-fetching ${label}`);
    }
  }
  return await new Response(progressStream(resp.body, label, total)).blob();
}

async function fetchBytes(url, label) {
  const blob = await fetchBlobCached(url, label);
  return new Uint8Array(await blob.arrayBuffer());
}

async function createSession(modelFile, dataFile, executionProviders, label) {
  post({ type: "status", detail: `loading ${label}` });
  const modelBytes = await fetchBytes(C.BASE + modelFile, modelFile);
  const opts = { executionProviders, graphOptimizationLevel: "all" };
  if (dataFile) {
    const blob = await fetchBlobCached(C.BASE + dataFile, dataFile);
    opts.externalData = [{ path: dataFile, data: blob }];
  }
  try {
    return await ort.InferenceSession.create(modelBytes, opts);
  } catch (err) {
    throw new Error(
      `failed to create ${label} session (${executionProviders
        .map((e) => e.name || e)
        .join(",")}): ${(err && err.message) || err}`,
    );
  }
}

async function init() {
  if (ready) {
    post({ type: "ready", encoderEP: ENC_EP });
    return;
  }
  if (initInFlight) return initInFlight;

  initInFlight = (async () => {
    const hasGPU = !!(self.navigator && navigator.gpu);

    post({ type: "status", detail: "fetching vocab" });
    const vb = await fetchBytes(C.BASE + "vocab.txt", "vocab.txt");
    VOCAB = new TextDecoder().decode(vb).split("\n");
    MELFB = buildMelFB();
    WINDOW = buildWindow();

    DEC = await createSession(
      "decoder.onnx",
      "decoder.onnx.data",
      ["wasm"],
      "decoder (CPU)",
    );
    JOINT = await createSession(
      "joint.onnx",
      "joint.onnx.data",
      ["wasm"],
      "joint (CPU)",
    );

    if (!hasGPU && IS_MOBILE) {
      throw new Error(
        "WebGPU is required on mobile for this 690 MB model, but navigator.gpu isn't available in this browser. Try the latest Chrome on Android, or Safari 18+ on iOS.",
      );
    }
    ENC_EP = hasGPU ? "webgpu" : "wasm";
    post({ type: "ep", ep: ENC_EP, encoder: true });
    try {
      ENC = await createSession(
        "encoder.onnx",
        "encoder.onnx.data",
        [{ name: ENC_EP }],
        `encoder (~690 MB, ${ENC_EP})`,
      );
    } catch (err) {
      if (ENC_EP === "webgpu" && !IS_MOBILE) {
        post({
          type: "ep",
          ep: "wasm",
          encoder: true,
          note: String((err && err.message) || err),
        });
        ENC_EP = "wasm";
        ENC = await createSession(
          "encoder.onnx",
          "encoder.onnx.data",
          ["wasm"],
          "encoder (~690 MB, wasm)",
        );
      } else {
        throw err;
      }
    }

    ready = true;
    post({ type: "ready", encoderEP: ENC_EP });
  })();

  try {
    await initInFlight;
  } catch (err) {
    initInFlight = null;
    throw err;
  }
  return initInFlight;
}

const f32 = (a, d) =>
  new ort.Tensor(
    "float32",
    a instanceof Float32Array ? a : Float32Array.from(a),
    d,
  );
const i64 = (a, d) =>
  new ort.Tensor("int64", BigInt64Array.from(a.map((v) => BigInt(v))), d);
const zeros = (n) => new Float32Array(n);

async function newState(langId) {
  const s = {
    langId,
    cch: zeros(C.LAYERS * 56 * C.D_MODEL),
    cct: zeros(C.LAYERS * C.D_MODEL * 8),
    ccl: 0,
    h: zeros(C.DEC_LAYERS * C.DEC_HID),
    c: zeros(C.DEC_LAYERS * C.DEC_HID),
    decOut: null,
    emitted: [],
  };
  await decoderStep(s, C.BLANK);
  return s;
}

async function decoderStep(s, token) {
  const r = await DEC.run({
    targets: i64([token], [1, 1]),
    h_in: f32(s.h, [C.DEC_LAYERS, 1, C.DEC_HID]),
    c_in: f32(s.c, [C.DEC_LAYERS, 1, C.DEC_HID]),
  });
  s.h = r.h_out.data;
  s.c = r.c_out.data;
  s.decOut = r.decoder_output.data;
}

async function jointArgmax(s, encFrame) {
  const r = await JOINT.run({
    encoder_output: f32(encFrame, [1, 1, C.D_MODEL]),
    decoder_output: f32(s.decOut, [1, 1, C.DEC_HID]),
  });
  const logits = r.joint_output.data;
  let best = 0;
  let bv = logits[0];
  for (let i = 1; i < C.VOCAB; i++) {
    if (logits[i] > bv) {
      bv = logits[i];
      best = i;
    }
  }
  return best;
}

async function encoderStep(s, buf65, length) {
  const er = await ENC.run({
    audio_signal: f32(buf65, [1, C.ENC_IN, C.N_MELS]),
    length: i64([length], [1]),
    cache_last_channel: f32(s.cch, [1, C.LAYERS, 56, C.D_MODEL]),
    cache_last_time: f32(s.cct, [1, C.LAYERS, C.D_MODEL, 8]),
    cache_last_channel_len: i64([s.ccl], [1]),
    lang_id: i64([s.langId], [1]),
  });
  const enc = er.outputs.data;
  const encT = er.outputs.dims[1];
  s.cch = er.cache_last_channel_next.data;
  s.cct = er.cache_last_time_next.data;
  s.ccl = Number(er.cache_last_channel_len_next.data[0]);
  for (let t = 0; t < encT; t++) {
    const fr = enc.subarray(t * C.D_MODEL, (t + 1) * C.D_MODEL);
    let sym = 0;
    while (sym < C.MAX_SYM) {
      const k = await jointArgmax(s, fr);
      if (k === C.BLANK) break;
      s.emitted.push(k);
      await decoderStep(s, k);
      sym++;
    }
  }
}

async function transcribeFull(samples, langId) {
  const mel = computeMelOffline(samples, MELFB, WINDOW);
  const s = await newState(langId);
  const steps = Math.ceil(mel.length / C.NEW_FRAMES);
  for (let step = 0; step < steps; step++) {
    const base = step * C.NEW_FRAMES;
    const buf = new Float32Array(C.ENC_IN * C.N_MELS);
    for (let i = 0; i < C.ENC_IN; i++) {
      const gi = base - C.CACHE_FRAMES + i;
      if (gi >= 0 && gi < mel.length) buf.set(mel[gi], i * C.N_MELS);
    }
    const validNew = Math.min(C.NEW_FRAMES, mel.length - base);
    await encoderStep(s, buf, C.CACHE_FRAMES + validNew);
    const { text, lang } = detok(s.emitted, VOCAB);
    post({
      type: "partial",
      text,
      lang,
      progress: (step + 1) / steps,
    });
  }
  const { text, lang } = detok(s.emitted, VOCAB);
  post({ type: "final", text, lang, tokens: s.emitted.length });
}

let stream = null;

async function streamStart(langId) {
  stream = {
    state: await newState(langId),
    mel: new StreamingMel(MELFB, WINDOW),
    frames: [],
    frameOffset: 0,
    consumed: 0,
  };
  post({ type: "stream-ready" });
}

async function streamAudio(samples) {
  if (!stream) return;
  const newFrames = stream.mel.push(samples);
  for (const fr of newFrames) stream.frames.push(fr);
  while (
    stream.frames.length + stream.frameOffset - stream.consumed >=
    C.NEW_FRAMES
  ) {
    await runStreamBlock(C.NEW_FRAMES);
    const { text, lang } = detok(stream.state.emitted, VOCAB);
    post({ type: "partial", text, lang });
  }
  post({ type: "stream-tick" });
}

async function runStreamBlock(validNew) {
  const s = stream;
  const base = s.consumed;
  const buf = new Float32Array(C.ENC_IN * C.N_MELS);
  for (let i = 0; i < C.ENC_IN; i++) {
    const gi = base - C.CACHE_FRAMES + i;
    const li = gi - s.frameOffset;
    if (gi >= 0 && li >= 0 && li < s.frames.length)
      buf.set(s.frames[li], i * C.N_MELS);
  }
  await encoderStep(s.state, buf, C.CACHE_FRAMES + validNew);
  s.consumed += C.NEW_FRAMES;
  const keepFrom = s.consumed - C.CACHE_FRAMES;
  if (keepFrom > s.frameOffset) {
    s.frames.splice(0, keepFrom - s.frameOffset);
    s.frameOffset = keepFrom;
  }
}

async function streamEnd() {
  if (!stream) return;
  const remaining = stream.frames.length + stream.frameOffset - stream.consumed;
  if (remaining > 0) {
    await runStreamBlock(Math.min(C.NEW_FRAMES, remaining));
  }
  const { text, lang } = detok(stream.state.emitted, VOCAB);
  post({
    type: "final",
    text,
    lang,
    tokens: stream.state.emitted.length,
  });
  stream = null;
}

async function clearCache() {
  try {
    await caches.delete(CACHE_NAME);
    post({ type: "status", detail: "cached model cleared" });
  } catch (e) {
    post({
      type: "status",
      detail: `clear cache failed: ${(e && e.name) || e}`,
    });
  }
}

async function handle(m) {
  switch (m.type) {
    case "init":
      await init();
      break;
    case "transcribeFull":
      await transcribeFull(new Float32Array(m.samples), m.langId);
      break;
    case "streamStart":
      await streamStart(m.langId);
      break;
    case "streamAudio":
      await streamAudio(new Float32Array(m.samples));
      break;
    case "streamEnd":
      await streamEnd();
      break;
    case "clearCache":
      await clearCache();
      break;
  }
}

let chain = Promise.resolve();
self.onmessage = (e) => {
  chain = chain
    .then(() => handle(e.data))
    .catch((err) => {
      stream = null;
      post({
        type: "error",
        message: (err && (err.stack || err.message)) || String(err),
      });
    });
};

self.onerror = (msg) =>
  post({ type: "error", message: `worker error: ${msg}` });

self.onunhandledrejection = (e) =>
  post({
    type: "error",
    message: `unhandled: ${(e && e.reason && (e.reason.message || e.reason)) || "unknown"}`,
  });
