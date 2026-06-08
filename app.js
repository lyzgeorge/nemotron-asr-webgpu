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
  diag: $("diag"), diagToggle: $("diagToggle"),
  // record
  recBtn: $("recBtn"), recTimer: $("recTimer"), recLevel: $("recLevelFill"),
  // live
  liveBtn: $("liveBtn"), liveLevel: $("liveLevelFill"), liveDot: $("liveDot"),
  // file
  file: $("fileInput"), fileBtn: $("fileBtn"),
};

/* ── worker ── */
const worker = new Worker("./worker.js", { type: "module" });
let readyResolve, readyPromise = null;

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

worker.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case "progress": {
      const pct = m.total ? ((m.loaded / m.total) * 100).toFixed(0) + "%" : (m.loaded / 1048576).toFixed(0) + " MB";
      setStatus(`loading ${m.label} · ${pct}`, "loading");
      break;
    }
    case "status": log(`· ${m.detail}`, "dim"); break;
    case "ready": setStatus("model ready", "online"); log("models ready", "ok"); readyResolve && readyResolve(); break;
    case "stream-ready": log("stream started", "ok"); break;
    case "partial": setTranscript(m.text, m.lang); if (m.progress != null) setStatus(`transcribing · ${(m.progress * 100).toFixed(0)}%`, "loading"); break;
    case "final": setTranscript(m.text, m.lang); setStatus("model ready", "online"); log(`done — ${m.tokens} tokens`, "ok"); finishBusy(); break;
    case "error": setStatus("error", "error"); log("ERROR: " + m.message, "err"); finishBusy(); break;
  }
};

function ensureReady() {
  if (!readyPromise) { readyPromise = new Promise((r) => (readyResolve = r)); setStatus("loading model…", "loading"); worker.postMessage({ type: "init" }); }
  return readyPromise;
}
const langId = () => parseInt(els.lang.value || "0", 10);

/* ── busy gate (prevents overlapping full jobs) ── */
let busy = false;
function startBusy() { busy = true; }
function finishBusy() { busy = false; }

/* ── WebGPU check ── */
if (!navigator.gpu) { els.warn.classList.remove("hidden"); setStatus("WebGPU unavailable", "error"); }

/* ── mode switching ── */
function setMode(mode) {
  els.modeBtns.forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  for (const k in els.panels) els.panels[k].classList.toggle("hidden", k !== mode);
}
els.modeBtns.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
setMode("file");
els.diagToggle.addEventListener("click", () => els.diag.classList.toggle("hidden"));

/* ── shared audio plumbing for mic (record + live) ── */
let audioCtx = null, micNode = null, micStream = null, micSource = null;
let onFrame = null; // current frame consumer

async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  audioCtx = new AudioContext({ sampleRate: SR });
  if (audioCtx.state === "suspended") await audioCtx.resume();
  await audioCtx.audioWorklet.addModule("./mic-processor.js");
  micSource = audioCtx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(audioCtx, "mic-processor");
  micNode.port.onmessage = (e) => onFrame && onFrame(e.data);
  micSource.connect(micNode); // not connected to destination → no monitoring/feedback
}
async function stopMic() {
  onFrame = null;
  try { micSource && micSource.disconnect(); micNode && micNode.disconnect(); } catch {}
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) await audioCtx.close();
  audioCtx = micNode = micStream = micSource = null;
}
function rms(frame) { let s = 0; for (let i = 0; i < frame.length; i++) s += frame[i] * frame[i]; return Math.sqrt(s / frame.length); }
function setLevel(el, frame) { const v = Math.min(1, rms(frame) * 4); el.style.width = (v * 100).toFixed(0) + "%"; }

/* ── FILE mode ── */
els.file.addEventListener("change", () => (els.fileBtn.disabled = !els.file.files.length));
els.fileBtn.addEventListener("click", async () => {
  if (busy || !els.file.files.length) return;
  startBusy(); els.fileBtn.disabled = true;
  try {
    await ensureReady();
    setStatus("decoding audio…", "loading");
    const file = els.file.files[0], ab = await file.arrayBuffer();
    const tmp = new AudioContext(); const dec = await tmp.decodeAudioData(ab); await tmp.close();
    const off = new OfflineAudioContext(1, Math.ceil(dec.duration * SR), SR);
    const src = off.createBufferSource(); src.buffer = dec; src.connect(off.destination); src.start();
    const rendered = await off.startRendering();
    const samples = rendered.getChannelData(0).slice();
    log(`file: ${file.name} — ${(samples.length / SR).toFixed(2)} s`, "dim");
    setStatus("transcribing…", "loading");
    worker.postMessage({ type: "transcribeFull", samples: samples.buffer, langId: langId() }, [samples.buffer]);
  } catch (err) { setStatus("error", "error"); log("ERROR: " + (err.stack || err.message), "err"); finishBusy(); }
  els.fileBtn.disabled = false;
});

/* ── RECORD mode (capture, then transcribe on stop) ── */
let recording = false, recBuffers = [], recStart = 0, recTimerId = null;
els.recBtn.addEventListener("click", async () => {
  if (!recording) {
    if (busy) return;
    try {
      await ensureReady(); await startMic();
      recBuffers = []; recording = true; recStart = performance.now();
      onFrame = (frame) => { recBuffers.push(frame); setLevel(els.recLevel, frame); };
      els.recBtn.textContent = "■ Stop"; els.recBtn.classList.replace("is-start", "is-stop");
      els.recTimer.textContent = "0.0s";
      recTimerId = setInterval(() => (els.recTimer.textContent = ((performance.now() - recStart) / 1000).toFixed(1) + "s"), 100);
      setStatus("recording…", "loading");
    } catch (err) { log("mic error: " + err.message, "err"); setStatus("mic error", "error"); }
  } else {
    recording = false; clearInterval(recTimerId); await stopMic();
    els.recBtn.textContent = "● Record"; els.recBtn.classList.replace("is-stop", "is-start");
    els.recLevel.style.width = "0%";
    const total = recBuffers.reduce((n, f) => n + f.length, 0);
    if (total === 0) { setStatus("model ready", "online"); return; }
    const samples = new Float32Array(total); let o = 0; for (const f of recBuffers) { samples.set(f, o); o += f.length; }
    recBuffers = [];
    log(`recorded ${(samples.length / SR).toFixed(2)} s`, "dim");
    startBusy(); setStatus("transcribing…", "loading");
    worker.postMessage({ type: "transcribeFull", samples: samples.buffer, langId: langId() }, [samples.buffer]);
  }
});

/* ── LIVE mode (real-time streaming) ── */
let live = false, liveBatch = [], liveBatchLen = 0;
const LIVE_FLUSH = 1600; // ~100 ms per worker message
els.liveBtn.addEventListener("click", async () => {
  if (!live) {
    if (busy) return;
    try {
      await ensureReady(); await startMic();
      worker.postMessage({ type: "streamStart", langId: langId() });
      setTranscript("", null);
      live = true; liveBatch = []; liveBatchLen = 0;
      onFrame = (frame) => {
        setLevel(els.liveLevel, frame);
        liveBatch.push(frame); liveBatchLen += frame.length;
        if (liveBatchLen >= LIVE_FLUSH) {
          const buf = new Float32Array(liveBatchLen); let o = 0; for (const f of liveBatch) { buf.set(f, o); o += f.length; }
          liveBatch = []; liveBatchLen = 0;
          worker.postMessage({ type: "streamAudio", samples: buf.buffer }, [buf.buffer]);
        }
      };
      els.liveBtn.textContent = "■ Stop"; els.liveBtn.classList.replace("is-start", "is-stop");
      els.liveDot.classList.add("online"); setStatus("listening…", "loading");
    } catch (err) { log("mic error: " + err.message, "err"); setStatus("mic error", "error"); }
  } else {
    live = false; await stopMic();
    if (liveBatchLen > 0) { const buf = new Float32Array(liveBatchLen); let o = 0; for (const f of liveBatch) { buf.set(f, o); o += f.length; } worker.postMessage({ type: "streamAudio", samples: buf.buffer }, [buf.buffer]); }
    liveBatch = []; liveBatchLen = 0;
    worker.postMessage({ type: "streamEnd" });
    els.liveBtn.textContent = "● Start listening"; els.liveBtn.classList.replace("is-stop", "is-start");
    els.liveDot.classList.remove("online"); els.liveLevel.style.width = "0%";
  }
});

els.stripTag.addEventListener("change", () => setTranscript(lastText, lastLang));
log("Ready. Pick a mode. First run downloads ~690 MB (cached after).", "dim");
log("Serve over http://localhost (module worker + mic require it — file:// won't work).", "dim");
