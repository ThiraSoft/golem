package main

// POST /v1/images/generations: a picture from a sentence, with Krea 2.
//
// It is OpenAI's endpoint with what the ComfyUI mobile front sends beside it:
// the negative prompt, the steps, the guidance and the seed. The answer is the
// picture as a base64 PNG, and the seed that drew it, so that a client that
// asked for a random one can ask for the same picture again. The PNG carries
// the request too, and the golem that drew it. A LoRA is named as the front
// names it, lora and lora_strength.

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"math/rand/v2"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ThiraSoft/golem/krea2"
)

// Imager draws pictures. krea2.Pipeline is one; the tests have another.
type Imager interface {
	Generate(r krea2.Request, progress krea2.Progress) (*image.NRGBA, krea2.Timings, error)
	WritePNG(w io.Writer, img image.Image, r krea2.Request) error
}

// lockedImager lets one picture be drawn at a time: the card holds one.
type lockedImager struct {
	mu sync.Mutex
	im Imager
}

func (l *lockedImager) WritePNG(w io.Writer, img image.Image, r krea2.Request) error {
	return l.im.WritePNG(w, img, r)
}

func (l *lockedImager) Generate(r krea2.Request, p krea2.Progress) (*image.NRGBA, krea2.Timings, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.im.Generate(r, p)
}

// SetImager lets this server draw pictures.
func (s *Server) SetImager(im Imager) { s.imager = &lockedImager{im: im} }

type generationRequest struct {
	Prompt         string   `json:"prompt"`
	NegativePrompt string   `json:"negative_prompt"`
	N              int      `json:"n"`
	Size           string   `json:"size"`
	ResponseFormat string   `json:"response_format"`
	Steps          int      `json:"steps"`
	CFG            *float32 `json:"cfg"`
	Seed           *int64   `json:"seed"`
	// A LoRA by its file name in ComfyUI's models/loras, as the front's
	// lora_name, and its strength, 1 when not given.
	Lora         string   `json:"lora"`
	LoraStrength *float32 `json:"lora_strength"`
}

// The mobile front's defaults.
const (
	defaultSize  = "768x1024"
	defaultSteps = 8
)

func (s *Server) generations(w http.ResponseWriter, r *http.Request) {
	var req generationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", "the request body is not the JSON this endpoint reads: "+err.Error())
		return
	}
	k, err := req.krea2()
	if err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	img, tm, err := s.imager.Generate(k, nil)
	if err != nil {
		refuse(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	var buf bytes.Buffer
	if err := s.imager.WritePNG(&buf, img, k); err != nil {
		refuse(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	if rec, ok := w.(*recorder); ok {
		rec.reason = fmt.Sprintf("%dx%d, %d steps, seed %d, %s", k.Width, k.Height, k.Steps, k.Seed, tm.Total.Round(time.Millisecond))
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"created": time.Now().Unix(),
		"data": []map[string]any{{
			"b64_json":       base64.StdEncoding.EncodeToString(buf.Bytes()),
			"revised_prompt": req.Prompt,
			"seed":           k.Seed,
		}},
	})
}

// krea2 reads the request as Krea 2 wants it, refusing what it cannot do.
func (req generationRequest) krea2() (krea2.Request, error) {
	var k krea2.Request
	if strings.TrimSpace(req.Prompt) == "" {
		return k, fmt.Errorf("prompt is required")
	}
	if req.N > 1 {
		return k, fmt.Errorf("n is %d; this server draws one picture a request", req.N)
	}
	if req.ResponseFormat != "" && req.ResponseFormat != "b64_json" {
		return k, fmt.Errorf("response_format %q: only b64_json, there is nowhere to host a url", req.ResponseFormat)
	}
	size := req.Size
	if size == "" {
		size = defaultSize
	}
	ws, hs, ok := strings.Cut(size, "x")
	width, err1 := strconv.Atoi(ws)
	height, err2 := strconv.Atoi(hs)
	if !ok || err1 != nil || err2 != nil {
		return k, fmt.Errorf("size %q is not WIDTHxHEIGHT", req.Size)
	}
	k = krea2.Request{Prompt: req.Prompt, Negative: req.NegativePrompt, Width: width, Height: height, Steps: req.Steps, CFG: 1}
	if k.Steps == 0 {
		k.Steps = defaultSteps
	}
	if req.CFG != nil {
		k.CFG = *req.CFG
	}
	if req.Lora != "" {
		if req.Lora != filepath.Base(req.Lora) || strings.HasPrefix(req.Lora, ".") {
			return k, fmt.Errorf("lora %q: a file name in the LoRA directory", req.Lora)
		}
		k.Lora, k.LoraStrength = req.Lora, 1
		if req.LoraStrength != nil {
			k.LoraStrength = *req.LoraStrength
		}
		if k.LoraStrength < 0 || k.LoraStrength > krea2.MaxLoRAStrength {
			return k, fmt.Errorf("lora_strength %g: from 0 to %d", k.LoraStrength, krea2.MaxLoRAStrength)
		}
	}
	if req.Seed != nil && *req.Seed >= 0 {
		k.Seed = uint64(*req.Seed)
	} else {
		// ComfyUI's seeds are under 2^50 from the front; so are these, so
		// that a client storing one as a JavaScript number keeps it whole.
		k.Seed = rand.Uint64N(1 << 50)
	}
	return k, nil
}
