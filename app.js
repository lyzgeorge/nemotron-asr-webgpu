// app.js — main thread: UI + audio capture + worker messaging. No model code here.
import { CONFIG } from "./shared.js";
const SR = CONFIG.SR;

const $ = (id) => document.getElementById(id);
const els = {
  warn: $("webgpuWarning"),
  modeBtns: [...document.querySelectorAll(".mode-btn")],
  panels: { record: $("panel-record"), live: $("panel-live"), file: $("panel-file") },
  lang: $("langId"), langChip: $("langChip"), stripTag: $("stripTag"),
  transcript: $("transcript"),
  statusDot: $("statusDot"), statusText: $("statusText"),
  diag: $("diag"), diagToggle: $("diagToggle"), clearCacheBtn: $("clearCacheBtn"),
  // record
  recBtn: $("recBtn"), recTimer: $("recTimer"), recLevel: $("recLevelFill"),
  // live
  liveBtn: $("liveBtn"), liveLevel: $("liveLevelFill"), liveDot: $("liveDot"),
  // file
  file: $("fileInput"), fileBtn: $("fileBtn"),
};

/* ── worker ── */
const worker = new Worker("./worker.js", { type: "module" });

function log(s, cls) { const e = document.createElement("span"); if (cls) e.className = cls; e.textContent = s + "\n"; els.diag.appendChild(e); els.diag.scrollTop = els.diag.scrollHeight; }
function setStatus(text, state) { els.statusText.textContent = text; els.statusDot.className = "status-dot" + (state ? " " + state : ""); }
let lastText = "", lastLang = null;
function setTranscript(text, lang) {
  lastText = text; lastLang = lang;
  const show = (els.stripTag.checked || !lang) ? text : (text + (lang ? ` <${lang}>` : ""));
  els.transcript.textContent = show || "—";
  els.transcript.classList.toggle("is-empty", !show);
  els.langChip.textContent = lang ? lang : ""; els.langChip.classList.toggle("hidden", !lang);
}

/* ── model-ready gate: resolves on "ready", rejects on a load "error" so callers
   can recover (and the user can retry) instead of hanging forever. ── */
let readyState = "idle"; // idle | loading | ready
let readyPromise = null, readyResolve = null, readyReject = null;
function ensureReady() {
  if (readyState === "ready") return Promise.resolve();
  if (readyState === "loading") return readyPromise;
  readyState = "loading";
  readyPromise = new Promise((res, rej) => { readyResolve = res; readyReject = rej; });
  setStatus("loading model…", "loading");
  worker.postMessage({ type: "init" });
  return readyPromise;
}

worker.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case "progress": {
      const pct = m.cached ? "cached" : m.total ? ((m.loaded / m.total) * 100).toFixed(0) + "%" : (m.loaded / 1048576).toFixed(0) + " MB";
      setStatus(`loading ${m.label} · ${pct}`, "loading");
      break;
    }
    case "status": log(`· ${m.detail}`, "dim"); break;
    case "ep": log(`${m.encoder ? "encoder " : ""}execution provider: ${m.ep}${m.ep === "wasm" ? " (slow — live may lag)" : ""}`, m.ep === "wasm" ? "err" : "ok"); if (m.note) log("webgpu fallback: " + m.note, "dim"); break;
    case "ready":
      readyState = "ready"; setStatus("model ready", "online");
      log(`models ready (encoder on ${m.encoderEP || "webgpu"})`, "ok");
      readyResolve && readyResolve();
      break;
    case "stream-ready": liveStreaming = true; log("stream started", "ok"); break;
    case "stream-tick": pendingBlocks = Math.max(0, pendingBlocks - 1); break;
    case "partial":
      setTranscript(m.text, m.lang);
      if (m.progress != null) setStatus(`transcribing · ${(m.progress * 100).toFixed(0)}%`, "loading");
      break;
    case "final": setTranscript(m.text, m.lang); setStatus("model ready", "online"); log(`done — ${m.tokens} tokens`, "ok"); finishBusy(); break;
    case "error":
      setStatus("error", "error"); log("ERROR: " + m.message, "err");
      // If the failure happened during load, fail the gate so callers unwind and a retry is possible.
      if (readyState === "loading") { readyState = "idle"; readyReject && readyReject(new Error(m.message)); }
      finishBusy(); stopLiveUI(); finishRecordUI();
      break;
  }
};

const langId = () => parseInt(els.lang.value || "0", 10);

/* ── busy gate (prevents overlapping full jobs) ── */
let busy = false;
function startBusy() { busy = true; }
function finishBusy() { busy = false; }

/* ── WebGPU check ── */
const HAS_GPU = !!navigator.gpu;
const IS_MOBILE = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent || "");
if (!HAS_GPU) {
  els.warn.classList.remove("hidden");
  els.warn.textContent = IS_MOBILE
    ? "WebGPU isn't available in this browser. This 690 MB model needs WebGPU — use the latest Chrome (Android) or Safari 18+ (iOS)."
    : "WebGPU isn't available. Use Chrome or Edge (or enable the WebGPU flag). A CPU fallback will work but is very slow.";
  if (IS_MOBILE) setStatus("WebGPU unavailable", "error");
}

/* ── mode switching ── */
function setMode(mode) {
  els.modeBtns.forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  for (const k in els.panels) els.panels[k].classList.toggle("hidden", k !== mode);
  // Kick off the (gesture-free) model load as soon as the user shows intent to
  // use the mic, so it's likely ready by the time they press the button.
  if ((mode === "record" || mode === "live") && readyState === "idle" && (HAS_GPU || !IS_MOBILE)) {
    ensureReady().catch(() => {});
  }
}
els.modeBtns.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
setMode("file");
els.diagToggle.addEventListener("click", () => els.diag.classList.toggle("hidden"));
els.clearCacheBtn.addEventListener("click", () => { worker.postMessage({ type: "clearCache" }); log("clearing cached model… reload the page to re-download", "dim"); });

/* ── shared audio plumbing for mic (record + live) ── */
let audioCtx = null, micNode = null, micStream = null, micSource = null;
let onFrame = null; // current frame consumer

async function startMic() {
  // Create the AudioContext synchronously inside the click handler so the user
  // gesture isn't "spent" by a slow await before it (mobile is strict about this).
  audioCtx = new AudioContext({ sampleRate: SR });
  micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  if (audioCtx.state === "suspended") await audioCtx.resume();
  await audioCtx.audioWorklet.addModule("./mic-processor.js");
  micSource = audioCtx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(audioCtx, "mic-processor");
  micNode.port.onmessage = (e) => onFrame && onFrame(e.data);
  micSource.connect(micNode);
  // Keep the node in the render graph so process() is pulled. The processor writes no
  // output, so the destination receives silence (no monitoring / no feedback).
  micNode.connect(audioCtx.destination);
  if (audioCtx.sampleRate !== SR) log(`note: AudioContext is ${audioCtx.sampleRate} Hz, expected ${SR} Hz — accuracy may suffer`, "err");
}
async function stopMic() {
  onFrame = null;
  try { micSource && micSource.disconnect(); micNode && micNode.disconnect(); } catch {}
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) { try { await audioCtx.close(); } catch {} }
  audioCtx = micNode = micStream = micSource = null;
}
function rms(frame) { let s = 0; for (let i = 0; i < frame.length; i++) s += frame[i] * frame[i]; return Math.sqrt(s / frame.length); }
function setLevel(el, frame) { const v = Math.min(1, rms(frame) * 4); el.style.width = (v * 100).toFixed(0) + "%"; }
function micErrorMessage(err) {
  const n = err && err.name;
  if (n === "NotAllowedError" || n === "SecurityError") return "microphone permission denied";
  if (n === "NotFoundError") return "no microphone found";
  if (n === "NotReadableError") return "microphone is in use by another app";
  return (err && err.message) || String(err);
}

/* ── FILE mode ── */
els.file.addEventListener("change", () => (els.fileBtn.disabled = !els.file.files.length));
els.fileBtn.addEventListener("click", async () => {
  if (busy || !els.file.files.length) return;
  startBusy(); els.fileBtn.disabled = true;
  try {
    setStatus("decoding audio…", "loading");
    const file = els.file.files[0], ab = await file.arrayBuffer();
    const tmp = new AudioContext(); const dec = await tmp.decodeAudioData(ab); await tmp.close();
    const off = new OfflineAudioContext(1, Math.ceil(dec.duration * SR), SR);
    const src = off.createBufferSource(); src.buffer = dec; src.connect(off.destination); src.start();
    const rendered = await off.startRendering();
    const samples = rendered.getChannelData(0).slice();
    log(`file: ${file.name} — ${(samples.length / SR).toFixed(2)} s`, "dim");
    await ensureReady(); // after decode so the gesture-free load overlaps the decode
    setStatus("transcribing…", "loading");
    worker.postMessage({ type: "transcribeFull", samples: samples.buffer, langId: langId() }, [samples.buffer]);
  } catch (err) { setStatus("error", "error"); log("ERROR: " + (err.stack || err.message), "err"); finishBusy(); }
  els.fileBtn.disabled = !els.file.files.length;
});

/* ── RECORD mode (capture, then transcribe on stop) ── */
let recording = false, recBuffers = [], recStart = 0, recTimerId = null;
function finishRecordUI() {
  recording = false; if (recTimerId) { clearInterval(recTimerId); recTimerId = null; }
  els.recBtn.textContent = "● Record"; els.recBtn.classList.remove("is-stop"); els.recBtn.classList.add("is-start");
  els.recLevel.style.width = "0%"; els.recBtn.disabled = false;
}
els.recBtn.addEventListener("click", async () => {
  if (!recording) {
    if (busy) return;
    els.recBtn.disabled = true;
    try {
      await startMic(); // grab the mic within the gesture first
    } catch (err) { log("mic error: " + micErrorMessage(err), "err"); setStatus("mic error", "error"); els.recBtn.disabled = false; return; }
    try {
      recBuffers = []; recording = true; recStart = performance.now();
      onFrame = (frame) => { recBuffers.push(frame); setLevel(els.recLevel, frame); };
      els.recBtn.textContent = "■ Stop"; els.recBtn.classList.replace("is-start", "is-stop"); els.recBtn.disabled = false;
      els.recTimer.textContent = "0.0s";
      recTimerId = setInterval(() => (els.recTimer.textContent = ((performance.now() - recStart) / 1000).toFixed(1) + "s"), 100);
      setStatus("recording…", "loading");
      ensureReady().catch((err) => log("model load failed: " + err.message, "err")); // load while recording
    } catch (err) { log("record error: " + (err.message || err), "err"); await stopMic(); finishRecordUI(); }
  } else {
    clearInterval(recTimerId); recTimerId = null; recording = false; await stopMic();
    els.recBtn.textContent = "● Record"; els.recBtn.classList.replace("is-stop", "is-start");
    els.recLevel.style.width = "0%";
    const total = recBuffers.reduce((n, f) => n + f.length, 0);
    if (total === 0) { setStatus("model ready", "online"); return; }
    const samples = new Float32Array(total); let o = 0; for (const f of recBuffers) { samples.set(f, o); o += f.length; }
    recBuffers = [];
    log(`recorded ${(samples.length / SR).toFixed(2)} s`, "dim");
    startBusy(); setStatus("transcribing…", "loading");
    try {
      await ensureReady();
      worker.postMessage({ type: "transcribeFull", samples: samples.buffer, langId: langId() }, [samples.buffer]);
    } catch (err) { setStatus("error", "error"); log("ERROR: " + (err.message || err), "err"); finishBusy(); }
  }
});

/* ── LIVE mode (real-time streaming) ── */
let live = false, liveStreaming = false, liveBatch = [], liveBatchLen = 0, pendingBlocks = 0;
const LIVE_FLUSH = 3200;          // ~200 ms per worker message (less postMessage churn)
const MAX_PENDING_BLOCKS = 24;    // cap un-decoded audio backlog → drop input rather than balloon latency
function stopLiveUI() {
  live = false; liveStreaming = false;
  els.liveBtn.textContent = "● Start listening"; els.liveBtn.classList.remove("is-stop"); els.liveBtn.classList.add("is-start");
  els.liveDot.classList.remove("online"); els.liveLevel.style.width = "0%"; els.liveBtn.disabled = false;
}
function postLiveChunk() {
  if (liveBatchLen === 0) return;
  const buf = new Float32Array(liveBatchLen); let o = 0; for (const f of liveBatch) { buf.set(f, o); o += f.length; }
  liveBatch = []; liveBatchLen = 0;
  if (pendingBlocks >= MAX_PENDING_BLOCKS) { log("dropping audio — decode can't keep up", "err"); return; }
  pendingBlocks++;
  worker.postMessage({ type: "streamAudio", samples: buf.buffer }, [buf.buffer]);
}
els.liveBtn.addEventListener("click", async () => {
  if (!live) {
    if (busy) return;
    els.liveBtn.disabled = true;
    try {
      await startMic(); // gesture-preserving mic start
    } catch (err) { log("mic error: " + micErrorMessage(err), "err"); setStatus("mic error", "error"); els.liveBtn.disabled = false; return; }

    live = true; liveStreaming = false; liveBatch = []; liveBatchLen = 0; pendingBlocks = 0;
    setTranscript("", null);
    els.liveBtn.textContent = "■ Stop"; els.liveBtn.classList.replace("is-start", "is-stop"); els.liveBtn.disabled = false;
    els.liveDot.classList.add("online"); setStatus("loading model…", "loading");
    // Until the stream is live, only animate the level meter — pre-ready audio is
    // dropped (no point transcribing audio captured during a multi-second load).
    onFrame = (frame) => {
      setLevel(els.liveLevel, frame);
      if (!liveStreaming) return;
      liveBatch.push(frame); liveBatchLen += frame.length;
      if (liveBatchLen >= LIVE_FLUSH) postLiveChunk();
    };
    try {
      await ensureReady();
      if (!live) return; // user stopped during load
      setStatus("listening…", "loading");
      worker.postMessage({ type: "streamStart", langId: langId() }); // "stream-ready" flips liveStreaming on
    } catch (err) {
      log("model load failed: " + err.message, "err"); setStatus("error", "error"); await stopMic(); stopLiveUI();
    }
  } else {
    const wasStreaming = liveStreaming;
    live = false; liveStreaming = false; await stopMic();
    if (wasStreaming) { postLiveChunk(); worker.postMessage({ type: "streamEnd" }); }
    liveBatch = []; liveBatchLen = 0;
    stopLiveUI();
  }
});

els.stripTag.addEventListener("change", () => setTranscript(lastText, lastLang));
log("Ready. Pick a mode. First run downloads ~750 MB (cached on-device after — re-open is instant).", "dim");
log("Serve over http://localhost or https:// (module worker + mic require it — file:// won't work).", "dim");
