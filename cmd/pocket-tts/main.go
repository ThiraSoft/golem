// Command pocket-tts synthesizes text into a WAV file.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/audio/wav"
	"github.com/ThiraSoft/golem/pockettts"
)

func main() {
	language := flag.String("language", orEnv("POCKET_TTS_LANGUAGE", pockettts.DefaultLanguage),
		"model language: "+strings.Join(pockettts.Languages(), ", "))
	weights := flag.String("weights", "", "model.safetensors; found in the Hugging Face cache when empty")
	tokenizer := flag.String("tokenizer", "", "tokenizer.model; found in the Hugging Face cache when empty")
	voice := flag.String("voice", os.Getenv("POCKET_TTS_VOICE"), "precomputed voice state (.safetensors)")
	clone := flag.String("clone", "", "clone the voice of this recording (.wav, mono, 24 kHz, twenty to thirty seconds)")
	saveVoice := flag.String("save-voice", "", "write the cloned voice here, to be reused with -voice")
	out := flag.String("o", "out.wav", "WAV file to write, or - for standard output")
	seed := flag.Uint64("seed", 0, "seed of the random draw; 0 for a different voice every time")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s [options] <text>\n", filepath.Base(os.Args[0]))
		fmt.Fprintln(os.Stderr, "  Without text, the text is read from standard input.")
		flag.PrintDefaults()
	}
	flag.Parse()

	lang, err := pockettts.LookupLanguage(*language)
	if err != nil {
		fail(err)
	}
	if *weights == "" {
		if *weights = orEnv("POCKET_TTS_WEIGHTS", pockettts.Locate(lang.WeightsPath())); *weights == "" {
			fail(missing(lang, "weights", "-weights", "POCKET_TTS_WEIGHTS"))
		}
	}
	if *tokenizer == "" {
		if *tokenizer = orEnv("POCKET_TTS_TOKENIZER", pockettts.Locate(lang.TokenizerPath())); *tokenizer == "" {
			fail(missing(lang, "tokenizer", "-tokenizer", "POCKET_TTS_TOKENIZER"))
		}
	}

	// Cloning a voice and saving it is a job of its own: there is nothing to
	// say and nothing to listen to, so no text is asked for.
	onlyCloning := *clone != "" && *saveVoice != "" && len(flag.Args()) == 0

	text := strings.Join(flag.Args(), " ")
	if text == "" && !onlyCloning {
		raw, err := io.ReadAll(os.Stdin)
		if err != nil {
			fail(err)
		}
		text = strings.TrimSpace(string(raw))
	}
	if text == "" && !onlyCloning {
		flag.Usage()
		os.Exit(2)
	}
	if *voice == "" && *clone == "" {
		fail(fmt.Errorf("no voice: pass -voice or -clone, or set POCKET_TTS_VOICE"))
	}

	start := time.Now()
	engine, err := pockettts.Open(pockettts.Options{
		Weights: *weights, Tokenizer: *tokenizer, Language: lang.Name,
	})
	if err != nil {
		fail(err)
	}
	defer engine.Close()
	loading := time.Since(start)

	var v *pockettts.Voice
	if *clone != "" {
		start := time.Now()
		if v, err = engine.VoiceFromWAV(*clone); err != nil {
			fail(err)
		}
		fmt.Fprintf(os.Stderr, "voice cloned from %s in %v\n",
			filepath.Base(*clone), time.Since(start).Round(time.Millisecond))
		if *saveVoice != "" {
			if err := engine.SaveVoice(*saveVoice, v); err != nil {
				fail(err)
			}
			fmt.Fprintf(os.Stderr, "voice written to %s — reuse it with -voice\n", *saveVoice)
		}
	} else if v, err = engine.LoadVoice(*voice); err != nil {
		fail(err)
	}

	if onlyCloning {
		return
	}

	start = time.Now()
	settings := pockettts.DefaultSettings(lang)
	settings.Seed = *seed
	sound, err := engine.Synthesize(text, v, &settings)
	if err != nil {
		fail(err)
	}
	elapsed := time.Since(start)

	var w io.Writer = os.Stdout
	if *out != "-" {
		f, err := os.Create(*out)
		if err != nil {
			fail(err)
		}
		defer f.Close()
		w = f
	}
	buffered := bufio.NewWriter(w)
	if err := wav.Write(buffered, sound, pockettts.SampleRate); err != nil {
		fail(err)
	}
	if err := buffered.Flush(); err != nil {
		fail(err)
	}

	seconds := float64(len(sound)) / pockettts.SampleRate
	fmt.Fprintf(os.Stderr, "loading %v — %.1f s of sound in %v, that is x%.2f real time\n",
		loading.Round(time.Millisecond), seconds, elapsed.Round(time.Millisecond),
		seconds/elapsed.Seconds())
}

// orEnv is the same for a plain string setting.
func orEnv(env, fallback string) string {
	if p := os.Getenv(env); p != "" {
		return p
	}
	return fallback
}

// missing says which language is not downloaded, rather than reporting an empty
// path. Only the language the daemon has fetched is present in the cache.
func missing(lang pockettts.Language, what, flagName, env string) error {
	return fmt.Errorf("no %s for %s in the Hugging Face cache; pass %s or set %s",
		what, lang.Name, flagName, env)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "pocket-tts:", err)
	os.Exit(1)
}
