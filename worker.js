// worker.js — all heavy lifting off the main thread.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.mjs";
import { CONFIG, buildMelFB, buildWindow, computeMelOffline, StreamingMel, detok } from "./shared.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.numThreads = 1;

const C = CONFIG;
const post = (m, t) => self.postMessage(m, t || []);

let ENC, DEC, JOINT, VOCAB, MELFB, WINDOW, ready = false;

/* ── loading ── */
async function fetchProg(url, label) {
  const r = await fetch(url); if (!r.ok) throw new Error(`HTTP ${r.status} ${label}`);
  const total = Number(r.headers.get("content-length")) || 0, rd = r.body.getReader();
  const chunks = []; let got = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; chunks.push(value); got += value.length; post({ type: "progress", label, loaded: got, total }); }
  const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out;
}
async function loadSession(model, data) {
  const opts = { executionProviders: ["webgpu"] };
  if (data) { const d = await fetchProg(C.BASE + data, data); opts.externalData = [{ path: data, data: d }]; }
  const m = await fetchProg(C.BASE + model, model);
  return ort.InferenceSession.create(m, opts);
}
async function init() {
  if (ready) { post({ type: "ready" }); return; }
  post({ type: "status", stage: "loading", detail: "fetching vocab" });
  VOCAB = (await (await fetch(C.BASE + "vocab.txt")).text()).split("\n");
  MELFB = buildMelFB(); WINDOW = buildWindow();
  post({ type: "status", stage: "loading", detail: "decoder" }); DEC = await loadSession("decoder.onnx", "decoder.onnx.data");
  post({ type: "status", stage: "loading", detail: "joint" }); JOINT = await loadSession("joint.onnx", "joint.onnx.data");
  post({ type: "status", stage: "loading", detail: "encoder (≈690 MB)" }); ENC = await loadSession("encoder.onnx", "encoder.onnx.data");
  ready = true; post({ type: "ready" });
}

/* ── tensor helpers ── */
const f32 = (a, d) => new ort.Tensor("float32", a instanceof Float32Array ? a : Float32Array.from(a), d);
const i64 = (a, d) => new ort.Tensor("int64", BigInt64Array.from(a.map((v) => BigInt(v))), d);
const zeros = (n) => new Float32Array(n);

/* ── per-utterance decode state ── */
async function newState(langId) {
  const s = {
    langId,
    cch: zeros(C.LAYERS * 56 * C.D_MODEL), cct: zeros(C.LAYERS * C.D_MODEL * 8), ccl: 0,
    h: zeros(C.DEC_LAYERS * C.DEC_HID), c: zeros(C.DEC_LAYERS * C.DEC_HID), decOut: null,
    emitted: [],
  };
  await decoderStep(s, C.BLANK); // SOS = blank
  return s;
}
async function decoderStep(s, token) {
  const r = await DEC.run({
    targets: i64([token], [1, 1]),
    h_in: f32(s.h, [C.DEC_LAYERS, 1, C.DEC_HID]),
    c_in: f32(s.c, [C.DEC_LAYERS, 1, C.DEC_HID]),
  });
  s.h = r.h_out.data; s.c = r.c_out.data;
  s.decOut = r.decoder_output.data; // [1,640,1] reused as [1,1,640]
}
async function jointArgmax(s, encFrame) {
  const r = await JOINT.run({
    encoder_output: f32(encFrame, [1, 1, C.D_MODEL]),
    decoder_output: f32(s.decOut, [1, 1, C.DEC_HID]),
  });
  const logits = r.joint_output.data; let best = 0, bv = logits[0];
  for (let i = 1; i < C.VOCAB; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
  return best;
}
// run one encoder step on a pre-assembled 65-frame buffer, then greedy-decode the emitted frames
async function encoderStep(s, buf65, length) {
  const er = await ENC.run({
    audio_signal: f32(buf65, [1, C.ENC_IN, C.N_MELS]),
    length: i64([length], [1]),
    cache_last_channel: f32(s.cch, [1, C.LAYERS, 56, C.D_MODEL]),
    cache_last_time: f32(s.cct, [1, C.LAYERS, C.D_MODEL, 8]),
    cache_last_channel_len: i64([s.ccl], [1]),
    lang_id: i64([s.langId], [1]),
  });
  const enc = er.outputs.data, encT = er.outputs.dims[1];
  s.cch = er.cache_last_channel_next.data; s.cct = er.cache_last_time_next.data;
  s.ccl = Number(er.cache_last_channel_len_next.data[0]);
  for (let t = 0; t < encT; t++) {
    const fr = enc.subarray(t * C.D_MODEL, (t + 1) * C.D_MODEL);
    let sym = 0;
    while (sym < C.MAX_SYM) {
      const k = await jointArgmax(s, fr);
      if (k === C.BLANK) break;
      s.emitted.push(k); await decoderStep(s, k); sym++;
    }
  }
}

/* ── full-audio (file / record) ── */
async function transcribeFull(samples, langId) {
  const mel = computeMelOffline(samples, MELFB, WINDOW);
  const s = await newState(langId);
  const steps = Math.ceil(mel.length / C.NEW_FRAMES);
  for (let step = 0; step < steps; step++) {
    const base = step * C.NEW_FRAMES, buf = new Float32Array(C.ENC_IN * C.N_MELS);
    for (let i = 0; i < C.ENC_IN; i++) { const gi = base - C.CACHE_FRAMES + i; if (gi >= 0 && gi < mel.length) buf.set(mel[gi], i * C.N_MELS); }
    const validNew = Math.min(C.NEW_FRAMES, mel.length - base);
    await encoderStep(s, buf, C.CACHE_FRAMES + validNew);
    const { text, lang } = detok(s.emitted, VOCAB);
    post({ type: "partial", text, lang, progress: (step + 1) / steps });
  }
  const { text, lang } = detok(s.emitted, VOCAB);
  post({ type: "final", text, lang, tokens: s.emitted.length });
}

/* ── live streaming ── */
let stream = null; // { state, mel: StreamingMel, frames: [], frameOffset, consumed }
async function streamStart(langId) {
  stream = { state: await newState(langId), mel: new StreamingMel(MELFB, WINDOW), frames: [], frameOffset: 0, consumed: 0 };
  post({ type: "stream-ready" });
}
async function streamAudio(samples) {
  if (!stream) return;
  const newFrames = stream.mel.push(samples);
  for (const fr of newFrames) stream.frames.push(fr);
  // process whenever ≥56 new frames are available
  while (stream.frames.length + stream.frameOffset - stream.consumed >= C.NEW_FRAMES) {
    await runStreamBlock(C.NEW_FRAMES);
    const { text, lang } = detok(stream.state.emitted, VOCAB);
    post({ type: "partial", text, lang });
  }
}
async function runStreamBlock(validNew) {
  const s = stream, base = s.consumed, buf = new Float32Array(C.ENC_IN * C.N_MELS);
  for (let i = 0; i < C.ENC_IN; i++) {
    const gi = base - C.CACHE_FRAMES + i, li = gi - s.frameOffset;
    if (gi >= 0 && li >= 0 && li < s.frames.length) buf.set(s.frames[li], i * C.N_MELS);
  }
  await encoderStep(s.state, buf, C.CACHE_FRAMES + validNew);
  s.consumed += C.NEW_FRAMES;
  const keepFrom = s.consumed - C.CACHE_FRAMES; // bound memory, retain 9-frame lookback
  if (keepFrom > s.frameOffset) { s.frames.splice(0, keepFrom - s.frameOffset); s.frameOffset = keepFrom; }
}
async function streamEnd() {
  if (!stream) return;
  const remaining = stream.frames.length + stream.frameOffset - stream.consumed;
  if (remaining > 0) { await runStreamBlock(Math.min(C.NEW_FRAMES, remaining)); }
  const { text, lang } = detok(stream.state.emitted, VOCAB);
  post({ type: "final", text, lang, tokens: stream.state.emitted.length });
  stream = null;
}

/* ── dispatch ── */
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") await init();
    else if (m.type === "transcribeFull") await transcribeFull(new Float32Array(m.samples), m.langId);
    else if (m.type === "streamStart") await streamStart(m.langId);
    else if (m.type === "streamAudio") await streamAudio(new Float32Array(m.samples));
    else if (m.type === "streamEnd") await streamEnd();
  } catch (err) {
    post({ type: "error", message: err && (err.stack || err.message) || String(err) });
  }
};
