# stt

Speech-to-text with the Kyutai STT model (`stt-1b-en_fr`).

Given audio in any common format (WAV, MP3, FLAC) at any sample rate and channel
count, this package transcribes what is said in English or French.

## Usage

```go
options, err := stt.Locate("/path/to/stt-checkpoint")
if err != nil {
    log.Fatal(err)
}
model, err := stt.Open(options)
if err != nil {
    log.Fatal(err)
}
defer model.Close()

// Whole-file transcription:
text, err := model.Transcribe(audioBytes)

// Streaming transcription:
err = model.TranscribeStream(ctx, audioBytes, func(seg stt.Segment) {
    fmt.Printf("[%5.2fs] %s\n", float64(seg.Frame)/12.5, seg.Text)
})
```

## Architecture

- **Codec**: Mimi SEANet encoder (24 kHz -> 12.5 Hz, latent dim 512) followed by a 32-level Split RVQ quantizer.
- **Trunk**: 16-layer autoregressive transformer (width 2048, 16 heads, SwiGLU FFN 5632, RoPE) decoding text tokens at 12.5 Hz.
