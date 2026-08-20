"""What the reference implementation costs, in the same terms as the Go benchmark.

Usage:
    python ref/bench_python.py [language] [voice] [runs]

Both sides are asked the same thing: one sentence, one voice, a fixed
end-of-speech threshold, and the ratio of sound produced to time taken. The
model is loaded and the voice prepared before the clock starts — a daemon pays
that once and speaks for hours — and the first synthesis is thrown away, because
the first pass through a lazily initialized graph is not what a speaking machine
does all day.

The Go side is `go test ./pockettts -run xxx -bench Synthesis`.
"""

import statistics
import sys
import time

from pocket_tts import TTSModel

SENTENCES = {
    "french_24l": "Le vent se lève, il faut tenter de vivre, et la mer scintille au loin.",
    "english_2026-01": "The wind is rising, we must try to live, and the sea glitters far away.",
}

# The threshold the Go engine defaults to, and what pocket_tts itself uses.
EOS_THRESHOLD = -4.0


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else "french_24l"
    voice = sys.argv[2] if len(sys.argv) > 2 else "alba"
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    sentence = SENTENCES[language]

    loading = time.perf_counter()
    model = TTSModel.load_model(language=language, eos_threshold=EOS_THRESHOLD)
    state = model.get_state_for_audio_prompt(voice)
    loading = time.perf_counter() - loading

    ratios, latencies, seconds = [], [], []
    for run in range(runs + 1):
        start = time.perf_counter()
        first, samples = None, 0
        for chunk in model.generate_audio_stream(state, sentence):
            if first is None:
                first = time.perf_counter() - start
            samples += chunk.numel()
        elapsed = time.perf_counter() - start
        audio = samples / model.sample_rate
        if run == 0:  # thrown away
            continue
        ratios.append(audio / elapsed)
        latencies.append(first * 1000)
        seconds.append(audio)

    print(f"language          {language}, voice {voice}")
    print(f"loaded in         {loading:.1f} s")
    print(f"audio per run     {statistics.median(seconds):.2f} s")
    print(f"x real time       {statistics.median(ratios):.2f}")
    print(f"to first chunk    {statistics.median(latencies):.0f} ms")


if __name__ == "__main__":
    main()
