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
		if *weights = orDefault("POCKET_TTS_WEIGHTS", cloningRepo, lang.WeightsPath()); *weights == "" {
			fail(missing(lang, "weights", "-weights", "POCKET_TTS_WEIGHTS"))
		}
	}
	if *tokenizer == "" {
		if *tokenizer = orDefault("POCKET_TTS_TOKENIZER", plainRepo, lang.TokenizerPath()); *tokenizer == "" {
			fail(missing(lang, "tokenizer", "-tokenizer", "POCKET_TTS_TOKENIZER"))
		}
	}

	text := strings.Join(flag.Args(), " ")
	if text == "" {
		raw, err := io.ReadAll(os.Stdin)
		if err != nil {
			fail(err)
		}
		text = strings.TrimSpace(string(raw))
	}
	if text == "" {
		flag.Usage()
		os.Exit(2)
	}
	if *voice == "" {
		fail(fmt.Errorf("no voice: pass -voice, or set POCKET_TTS_VOICE"))
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

	v, err := engine.LoadVoice(*voice)
	if err != nil {
		fail(err)
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

// The two Hugging Face repositories the Python daemon downloads into. The
// weights come from the voice-cloning one, the tokenizer from the other; that
// is the split the upstream configs use.
const cloningRepo = ".cache/huggingface/hub/models--kyutai--pocket-tts/snapshots/*/"
const plainRepo = ".cache/huggingface/hub/models--kyutai--pocket-tts-without-voice-cloning/snapshots/*/"

// orDefault looks for the file where the Python daemon already downloaded it.
// A snapshot that does not hold the language simply yields nothing, and the
// engine then reports the missing file rather than guessing.
func orDefault(env, repo, relative string) string {
	if p := os.Getenv(env); p != "" {
		return p
	}
	found, _ := filepath.Glob(filepath.Join(os.Getenv("HOME"), repo, relative))
	if len(found) > 0 {
		return found[0]
	}
	return ""
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
