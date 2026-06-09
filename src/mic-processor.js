// mic-processor.js — runs on the audio rendering thread; forwards mono PCM frames to the main thread.
class MicProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) this.port.postMessage(ch.slice(0)); // copy: the input buffer is recycled
    return true;
  }
}
registerProcessor("mic-processor", MicProcessor);
