// Float samples from the microphone, sent on as 16-bit little-endian PCM in ~100 ms chunks.
class Pcm16 extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunk = new Int16Array(sampleRate / 10);
    this.filled = 0;
  }

  process(inputs) {
    const samples = inputs[0]?.[0];
    if (!samples) return true;
    for (let i = 0; i < samples.length; i += 1) {
      const clamped = Math.max(-1, Math.min(1, samples[i]));
      this.chunk[this.filled] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      this.filled += 1;
      if (this.filled === this.chunk.length) {
        this.port.postMessage(this.chunk.buffer.slice(0));
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm16", Pcm16);
