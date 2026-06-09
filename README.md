# Nemotron 3.5 ASR — in-browser (WebGPU)

NVIDIA **Nemotron 3.5 ASR** (cache-aware FastConformer-RNNT, INT4 ONNX) running **entirely client-side** via `onnxruntime-web` on WebGPU. No server, no API key — audio never leaves the browser. Model weights (~750 MB total; the encoder alone is ~690 MB) are fetched from the Hugging Face `onnx-community` repo on first use and **persisted on-device via the Cache API**, so re-opening the page is instant.

## Modes
- **Record** — capture from the mic, transcribe on stop (offline center=true features, highest accuracy).
- **Live** — real-time streaming; text appears as you speak (streaming features + cache-aware encoder loop).
- **File** — decode + resample any audio file to 16 kHz locally, then transcribe.

## Architecture
| File | Role |
| --- | --- |
| `index.html` | markup only |
| `styles.css` | Spark Dark design system + ASR components |
| `shared.js` | constants + pure DSP (mel filterbank, FFT, offline/streaming features, detokenizer) |
| `worker.js` | Web Worker: ONNX sessions, model loading/caching, feature extraction, encoder loop, RNN-T greedy decode |
| `mic-processor.js` | AudioWorklet: forwards 16 kHz PCM off the main thread |
| `app.js` | main thread: UI, audio capture, worker messaging |

All heavy compute runs in the worker, so the UI never blocks.

## Execution-provider strategy
- **Encoder** (heavy conv/attention, ~690 MB) runs on **WebGPU**.
- **Decoder + joint** (tiny, run autoregressively per token) run on the **CPU (wasm)**. Running the RNN-T greedy loop on WebGPU would mean dozens-to-hundreds of tiny GPU dispatch/readback round-trips per audio chunk — the GPU-sync latency dominates and live streaming can't keep up. The CPU has no per-call sync cost, which is what makes real-time streaming work.

## Memory & mobile
- Weights stream **straight into the Cache API** (disk-backed) and are handed to ONNX Runtime as a `Blob`. We never build a 690 MB `ArrayBuffer` in the JS heap — that double-buffering (and `Response.arrayBuffer()` on very large files) is what crashed mobile tabs.
- Mobile **requires** WebGPU for the encoder; the 690 MB model is not run on the wasm CPU on phones (it OOMs / is far too slow). A clear message is shown if WebGPU is missing.
- "Clear cached model" (under diagnostics) wipes the on-device cache.

## Run locally
Module workers and the mic require a real origin — `file://` will **not** work.
```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy on Netlify
Static site, no build. `netlify.toml` sets `publish = "."`. Connect this repo in Netlify and deploy — it serves over HTTPS, which satisfies the WebGPU + microphone secure-context requirement.

## Requirements
- A WebGPU browser (Chrome/Edge desktop; Chrome on Android; Safari 18+ on iOS; Firefox behind a flag).
- ~750 MB download on first load (persisted on-device afterward via the Cache API).

## Notes
- `lang_id` defaults to `0` (auto). The model appends a detected `<xx-XX>` tag, shown as a chip and strippable via the checkbox.
- Model © NVIDIA; ONNX export by the Hugging Face `onnx-community`.
