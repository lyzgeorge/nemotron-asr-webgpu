# Nemotron 3.5 ASR — in-browser (WebGPU)

NVIDIA **Nemotron 3.5 ASR** (cache-aware FastConformer-RNNT, INT4 ONNX) running **entirely client-side** via `onnxruntime-web` on WebGPU. Model weights (~750 MB total; the encoder alone is ~690 MB) are fetched from the Hugging Face `onnx-community` repo and **persisted on-device via the Cache API**, so re-opening the page is instant.

Click **Load model** to start the one-time download — the Record / Live / File controls stay disabled until the model is ready.

## Modes
- **Record** — capture from the mic, transcribe on stop (offline `center=true` features, highest accuracy).
- **Live** — real-time streaming. A voice-activity detector gates the model so it only runs while you're speaking, and each speech segment becomes its own transcript line. Text appears as you talk.
- **File** — decode + resample any audio file to 16 kHz locally, then transcribe.

`lang_id` defaults to `0` (auto). The model appends a detected `<xx-XX>` language tag, shown as a chip and strippable via the checkbox.

## Project structure
```
.
├── index.html          # markup; loads src/app.js as a module
├── netlify.toml        # static deploy config (publish = ".")
├── README.md
└── src/
    ├── styles.css        # Spark Dark design system + ASR components
    ├── app.js            # main thread: UI, model-load gating, audio capture, VAD, worker messaging
    ├── worker.js         # Web Worker: ONNX sessions, model loading/caching, encoder loop, RNN-T greedy decode
    ├── shared.js         # constants + pure DSP (mel filterbank, FFT, offline/streaming features, detokenizer)
    └── mic-processor.js  # AudioWorklet: forwards 16 kHz PCM frames off the main thread
```

All heavy compute runs in the worker, so the UI never blocks.

## How Live mode works
- `mic-processor.js` frames the 16 kHz mic stream; the main thread reads those frames.
- An energy-based voice-activity detector (`createEnergyVAD` in `app.js`) with an adaptive noise floor and onset/hangover hysteresis decides when speech starts and stops.
- On speech **start** a streaming session opens (`streamStart`) and audio is sent to the worker in ~200 ms chunks while speech is active. On speech **end** the segment is finalized (`streamEnd`) and committed as its own transcript line.
- The streaming feature extractor (`StreamingMel` in `shared.js`) produces frames identical to the offline path (same `center=true` framing), so the encoder receives the features it was trained on.

## Execution-provider split
- **Encoder** (heavy conv/attention, ~690 MB) runs on **WebGPU**.
- **Decoder + joint** (tiny, run autoregressively per token) run on the **CPU (wasm)**. Running the RNN-T greedy loop on WebGPU would mean many tiny GPU dispatch/readback round-trips per audio chunk; the CPU has no per-call sync cost, which is what lets live streaming keep up.

## Memory & mobile
- Weights stream **straight into the Cache API** (disk-backed) and are handed to ONNX Runtime as a `Blob` — no 690 MB `ArrayBuffer` is built in the JS heap.
- Mobile **requires** WebGPU for the encoder; the 690 MB model is not run on the wasm CPU on phones. A clear message is shown when WebGPU is missing.
- **Clear cached model** (under diagnostics) wipes the on-device cache; reload to re-download.

## Install as an app (PWA)
The site ships a web app manifest and a service worker that caches the app shell, so it can be installed to the home screen and opened offline once the model has been downloaded (the weights live in their own Cache API bucket).
- **Android / Chrome:** open the site → menu → *Install app* (or *Add to Home screen*). WebGPU is required.
- **iPhone / Safari:** Share → *Add to Home Screen*. Needs iOS 26 (WebGPU on by default). Home-screen apps are exempt from Safari's 7-day storage eviction, but a 690 MB cache entry can still be evicted under storage pressure.
- Do the first load on Wi-Fi; nothing runs while the app is backgrounded or the screen is locked.

## Run locally
Module workers and the mic require a real origin — `file://` will **not** work.
```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy on Netlify
Static site, no build step. `netlify.toml` sets `publish = "."`. Connect the repo in Netlify and deploy — it serves over HTTPS, which satisfies the WebGPU + microphone secure-context requirement.

## Requirements
- A WebGPU browser (Chrome/Edge desktop; Chrome on Android; Safari 18+ on iOS; Firefox behind a flag).
- ~750 MB download on first load (persisted on-device afterward via the Cache API).

## Credits
Model © NVIDIA. ONNX export by the Hugging Face `onnx-community`. Inference via `onnxruntime-web`.
