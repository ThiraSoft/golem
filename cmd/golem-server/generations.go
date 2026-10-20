package main

// POST /v1/images/generations: a picture from a sentence, with Krea 2 or FLUX 3.
//
// It is OpenAI's endpoint with what the ComfyUI mobile front sends beside it:
// the negative prompt, the steps, the guidance, the seed and reference images.
// The answer is the picture as a base64 PNG, and the seed that drew it, so that
// a client that asked for a random one can ask for the same picture again. The
// PNG carries the request too, and the golem that drew it. A LoRA is named as the
// front names it, lora and lora_strength. hide_prompt keeps the prompt out of the
// PNG. With stream, the answer is server-sent events, one a step.

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math/rand/v2"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ThiraSoft/golem/flux3"
	"github.com/ThiraSoft/golem/krea2"
)

// PictureRequest carries what any pipeline needs to draw one picture.
type PictureRequest struct {
	Prompt         string
	Negative       string
	Width          int
	Height         int
	Steps          int
	CFG            float32
	Seed           uint64
	Lora           string
	LoraStrength   float32
	Chroma         string
	ChromaStrength float32
	ChromaSpread   float32
	HidePrompt     bool
	Images         []image.Image
}

// Imager draws pictures. krea2Imager and flux3Imager are two; the tests have another.
type Imager interface {
	Generate(r PictureRequest, progress krea2.Progress) (*image.NRGBA, krea2.Timings, error)
	WritePNG(w io.Writer, img image.Image, r PictureRequest) error
	Defaults() (size string, steps int)
	SupportsImages() bool
	SupportsLoRA() bool
}

func toKrea2(r PictureRequest) krea2.Request {
	return krea2.Request{
		Prompt:         r.Prompt,
		Negative:       r.Negative,
		Width:          r.Width,
		Height:         r.Height,
		Steps:          r.Steps,
		CFG:            r.CFG,
		Seed:           r.Seed,
		Lora:           r.Lora,
		LoraStrength:   r.LoraStrength,
		Chroma:         r.Chroma,
		ChromaStrength: r.ChromaStrength,
		ChromaSpread:   r.ChromaSpread,
		HidePrompt:     r.HidePrompt,
	}
}

func toFlux3(r PictureRequest) flux3.Request {
	return flux3.Request{
		Prompt:     r.Prompt,
		Negative:   r.Negative,
		Width:      r.Width,
		Height:     r.Height,
		Steps:      r.Steps,
		CFG:        r.CFG,
		Seed:       r.Seed,
		Images:     r.Images,
		HidePrompt: r.HidePrompt,
	}
}

type krea2Imager struct {
	p *krea2.Pipeline
}

func (ki *krea2Imager) Generate(r PictureRequest, progress krea2.Progress) (*image.NRGBA, krea2.Timings, error) {
	return ki.p.Generate(toKrea2(r), progress)
}

func (ki *krea2Imager) WritePNG(w io.Writer, img image.Image, r PictureRequest) error {
	return ki.p.WritePNG(w, img, toKrea2(r))
}

func (ki *krea2Imager) Defaults() (string, int) {
	return defaultSize, defaultSteps
}

func (ki *krea2Imager) SupportsImages() bool { return false }
func (ki *krea2Imager) SupportsLoRA() bool   { return true }

type flux3Imager struct {
	p *flux3.Pipeline
}

func (fi *flux3Imager) Generate(r PictureRequest, progress krea2.Progress) (*image.NRGBA, krea2.Timings, error) {
	return fi.p.Generate(toFlux3(r), progress)
}

func (fi *flux3Imager) WritePNG(w io.Writer, img image.Image, r PictureRequest) error {
	return fi.p.WritePNG(w, img, toFlux3(r))
}

func (fi *flux3Imager) Defaults() (string, int) {
	return "1024x1024", flux3.Defaults.Steps
}

func (fi *flux3Imager) SupportsImages() bool { return true }
func (fi *flux3Imager) SupportsLoRA() bool   { return false }

// lockedImager lets one picture be drawn at a time: the card holds one.
type lockedImager struct {
	mu sync.Mutex
	im Imager
}

func (l *lockedImager) WritePNG(w io.Writer, img image.Image, r PictureRequest) error {
	return l.im.WritePNG(w, img, r)
}

func (l *lockedImager) Generate(r PictureRequest, p krea2.Progress) (*image.NRGBA, krea2.Timings, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.im.Generate(r, p)
}

func (l *lockedImager) Defaults() (string, int) {
	return l.im.Defaults()
}

func (l *lockedImager) SupportsImages() bool {
	return l.im.SupportsImages()
}

func (l *lockedImager) SupportsLoRA() bool {
	return l.im.SupportsLoRA()
}

// SetImager lets this server draw pictures.
func (s *Server) SetImager(im Imager) { s.imager = &lockedImager{im: im} }

// SetKrea2 wires Krea 2 to answer picture generation requests.
func (s *Server) SetKrea2(p *krea2.Pipeline) { s.SetImager(&krea2Imager{p: p}) }

// SetFlux3 wires FLUX 3 to answer picture generation requests.
func (s *Server) SetFlux3(p *flux3.Pipeline) { s.SetImager(&flux3Imager{p: p}) }

type generationRequest struct {
	Prompt         string   `json:"prompt"`
	NegativePrompt string   `json:"negative_prompt"`
	N              int      `json:"n"`
	Size           string   `json:"size"`
	ResponseFormat string   `json:"response_format"`
	Steps          int      `json:"steps"`
	CFG            *float32 `json:"cfg"`
	Seed           *int64   `json:"seed"`
	Images         []string `json:"images"`
	// A LoRA by its file name in ComfyUI's models/loras, as the front's
	// lora_name, and its strength, 1 when not given.
	Lora         string   `json:"lora"`
	LoraStrength *float32 `json:"lora_strength"`
	// HidePrompt leaves the prompt out of the PNG's metadata.
	HidePrompt bool `json:"hide_prompt"`
	// Stream answers with server-sent events: one a step, then the picture.
	Stream bool `json:"stream"`
}

// The mobile front's defaults.
const (
	defaultSize  = "768x1024"
	defaultSteps = 8
)

func (s *Server) generations(w http.ResponseWriter, r *http.Request) {
	if s.imager == nil {
		refuse(w, http.StatusNotFound, "not_found", "this server does not draw pictures")
		return
	}
	var req generationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", "the request body is not the JSON this endpoint reads: "+err.Error())
		return
	}
	k, err := req.pictureRequest(s.imager)
	if err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if req.Stream {
		s.streamGeneration(w, req, k)
		return
	}
	png, err := s.draw(w, k, nil)
	if err != nil {
		refuse(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"created": time.Now().Unix(),
		"data": []map[string]any{{
			"b64_json":       base64.StdEncoding.EncodeToString(png),
			"revised_prompt": req.Prompt,
			"seed":           k.Seed,
		}},
	})
}

// draw draws the picture and returns it as a PNG, noting its cost for the
// log line.
func (s *Server) draw(w http.ResponseWriter, k PictureRequest, progress krea2.Progress) ([]byte, error) {
	img, tm, err := s.imager.Generate(k, progress)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := s.imager.WritePNG(&buf, img, k); err != nil {
		return nil, err
	}
	if rec, ok := w.(*recorder); ok {
		rec.reason = fmt.Sprintf("%dx%d, %d steps, seed %d, %s", k.Width, k.Height, k.Steps, k.Seed, tm.Total.Round(time.Millisecond))
	}
	return buf.Bytes(), nil
}

// streamGeneration answers with server-sent events, named as OpenAI names
// those of a streamed picture: image_generation.progress after each step
// (step, steps), then image_generation.completed with the picture, or error.
// A client waiting behind another picture hears nothing until its own starts.
func (s *Server) streamGeneration(w http.ResponseWriter, req generationRequest, k PictureRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		refuse(w, http.StatusInternalServerError, "server_error", "this connection cannot be streamed to")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	send := func(kind string, body map[string]any) {
		body["type"] = kind
		data, _ := json.Marshal(body)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", kind, data)
		flusher.Flush()
	}
	png, err := s.draw(w, k, func(step, steps int) {
		send("image_generation.progress", map[string]any{"step": step, "steps": steps})
	})
	if err != nil {
		send("error", map[string]any{"error": map[string]any{"type": "server_error", "message": err.Error()}})
		return
	}
	send("image_generation.completed", map[string]any{
		"created_at":     time.Now().Unix(),
		"b64_json":       base64.StdEncoding.EncodeToString(png),
		"revised_prompt": req.Prompt,
		"seed":           k.Seed,
	})
}

func decodeImageBase64(s string) (image.Image, error) {
	if idx := strings.Index(s, ","); idx != -1 && strings.HasPrefix(s, "data:") {
		s = s[idx+1:]
	}
	s = strings.TrimSpace(s)
	data, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("invalid base64 image: %w", err)
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image config: %w", err)
	}
	const maxDim = 4096
	if cfg.Width > maxDim || cfg.Height > maxDim {
		return nil, fmt.Errorf("image dimension %dx%d exceeds %dx%d limit", cfg.Width, cfg.Height, maxDim, maxDim)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

// pictureRequest reads the request, refusing what it cannot do.
func (req generationRequest) pictureRequest(im Imager) (PictureRequest, error) {
	var p PictureRequest
	if strings.TrimSpace(req.Prompt) == "" {
		return p, fmt.Errorf("prompt is required")
	}
	if req.N > 1 {
		return p, fmt.Errorf("n is %d; this server draws one picture a request", req.N)
	}
	if req.ResponseFormat != "" && req.ResponseFormat != "b64_json" {
		return p, fmt.Errorf("response_format %q: only b64_json, there is nowhere to host a url", req.ResponseFormat)
	}
	defSize, defSteps := defaultSize, defaultSteps
	if im != nil {
		defSize, defSteps = im.Defaults()
	}
	size := req.Size
	if size == "" {
		size = defSize
	}
	ws, hs, ok := strings.Cut(size, "x")
	width, err1 := strconv.Atoi(ws)
	height, err2 := strconv.Atoi(hs)
	if !ok || err1 != nil || err2 != nil {
		return p, fmt.Errorf("size %q is not WIDTHxHEIGHT", req.Size)
	}
	steps := req.Steps
	if steps == 0 {
		steps = defSteps
	}
	p = PictureRequest{
		Prompt:     req.Prompt,
		Negative:   req.NegativePrompt,
		Width:      width,
		Height:     height,
		Steps:      steps,
		CFG:        1,
		HidePrompt: req.HidePrompt,
	}
	if req.CFG != nil {
		p.CFG = *req.CFG
	}
	if req.Lora != "" {
		if im != nil && !im.SupportsLoRA() {
			return p, fmt.Errorf("lora is not supported by this model")
		}
		if req.Lora != filepath.Base(req.Lora) || strings.HasPrefix(req.Lora, ".") {
			return p, fmt.Errorf("lora %q: a file name in the LoRA directory", req.Lora)
		}
		p.Lora, p.LoraStrength = req.Lora, 1
		if req.LoraStrength != nil {
			p.LoraStrength = *req.LoraStrength
		}
		if p.LoraStrength < 0 || p.LoraStrength > krea2.MaxLoRAStrength {
			return p, fmt.Errorf("lora_strength %g: from 0 to %d", p.LoraStrength, krea2.MaxLoRAStrength)
		}
	}
	if len(req.Images) > 0 {
		if im != nil && !im.SupportsImages() {
			return p, fmt.Errorf("images is not supported by this model")
		}
		if len(req.Images) > 4 {
			return p, fmt.Errorf("images: at most 4 reference images, got %d", len(req.Images))
		}
		for i, raw := range req.Images {
			img, err := decodeImageBase64(raw)
			if err != nil {
				return p, fmt.Errorf("images[%d]: %w", i, err)
			}
			p.Images = append(p.Images, img)
		}
	}
	if req.Seed != nil && *req.Seed >= 0 {
		p.Seed = uint64(*req.Seed)
	} else {
		// ComfyUI's seeds are under 2^50 from the front; so are these, so
		// that a client storing one as a JavaScript number keeps it whole.
		p.Seed = rand.Uint64N(1 << 50)
	}
	return p, nil
}
