# Nemotron 3.5 ASR — in-browser (WebGPU)

NVIDIA **Nemotron 3.5 ASR** (cache-aware FastConformer-RNNT, INT4 ONNX) running **entirely client-side** via `onnxruntime-web` on WebGPU. No server, no API key — audio never leaves the browser. Model weights (~690 MB) are fetched from the Hugging Face `onnx-community` repo on first use and cached by the browser.

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
| `worker.js` | Web Worker: ONNX sessions (WebGPU), feature extraction, encoder loop, RNN-T greedy decode |
| `mic-processor.js` | AudioWorklet: forwards 16 kHz PCM off the main thread |
| `app.js` | main thread: UI, audio capture, worker messaging |

All heavy compute runs in the worker, so the UI never blocks.

## Run locally
Module workers and the mic require a real origin — `file://` will **not** work.
```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy on Netlify
Static site, no build. `netlify.toml` sets `publish = "."`. Connect this repo in Netlify and deploy — it serves over HTTPS, which satisfies the WebGPU + microphone secure-context requirement.

## Requirements
- A WebGPU browser (Chrome/Edge; Safari/Firefox behind a flag).
- ~690 MB download on first load (cached afterward).

## Notes
- `lang_id` defaults to `0` (auto). The model appends a detected `<xx-XX>` tag, shown as a chip and strippable via the checkbox.
- Model © NVIDIA; ONNX export by the Hugging Face `onnx-community`.
