package main

// Embeddings, under the two APIs a client already speaks: OpenAI's
// /v1/embeddings, and ollama's /api/embed with its older /api/embeddings. A
// client that was pointed at ollama is pointed here by changing the port.
//
// The three differ in small ways that clients depend on, and each is kept:
// OpenAI's client sends encoding_format base64 unless told otherwise and
// decodes it itself; ollama's /api/embed truncates a text that is too long
// unless told not to, and normalizes; its older /api/embeddings does neither.

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"time"

	"github.com/ThiraSoft/golem/nomic"
)

// Embedder is what the endpoints need of a model: framing a text, and a pass
// over several.
type Embedder interface {
	// Encode is the text as identifiers, framed as the model reads it and not
	// cut to the context.
	Encode(text string) []int32
	// Context is the most identifiers one text may have.
	Context() int
	// Embed is one pooled, unnormalized vector per text.
	Embed(texts [][]int32) ([][]float32, error)
}

// SetEmbedder lets this server answer embedding requests. Requests that arrive
// together are answered by one pass: see batcher.
func (s *Server) SetEmbedder(e Embedder) { s.embed = newBatcher(e) }

// batcher gathers the texts of requests that arrive while a pass is running
// into the next pass.
//
// A pass reads every weight of the model once for all the positions it
// carries, so sixty-four one-sentence requests answered one pass each read the
// model sixty-four times, and answered together read it once. Nothing waits
// for company: a request that finds the model idle goes at once, and only the
// ones that arrive during a pass are held — for no longer than that pass —
// and carried together. It is llama-server's continuous batching, for a model
// that has no generation to interleave. The one exception is a pass too
// small to be worth its fixed cost, which waits a few milliseconds for company:
// windowPositions says why.
type batcher struct {
	Embedder
	requests chan *embedJob
}

type embedJob struct {
	texts [][]int32
	vecs  [][]float32
	err   error
	done  chan struct{}
}

// batchPositions bounds what one gathered pass carries. The model cuts a
// longer one itself; this only stops one pass from taking every waiting
// request when a second pass would start sooner.
const batchPositions = 4096

// A pass that would carry fewer than windowPositions waits up to window for
// company. On the card a pass of one sentence costs six milliseconds and one
// of eight costs ten, so eight clients each sending a sentence at a time are
// served faster by holding the first of them a few milliseconds than by starting
// it alone and making the other seven wait out its whole pass.
const (
	windowPositions = 256
	window          = 3 * time.Millisecond
)

func newBatcher(e Embedder) *batcher {
	b := &batcher{Embedder: e, requests: make(chan *embedJob, 1024)}
	go b.run()
	return b
}

func (b *batcher) Embed(texts [][]int32) ([][]float32, error) {
	j := &embedJob{texts: texts, done: make(chan struct{})}
	b.requests <- j
	<-j.done
	return j.vecs, j.err
}

func (b *batcher) run() {
	var held *embedJob
	for {
		jobs := []*embedJob{held}
		if held == nil {
			jobs[0] = <-b.requests
		}
		held = nil
		positions := 0
		for _, t := range jobs[0].texts {
			positions += len(t)
		}
		var wait <-chan time.Time
		if positions < windowPositions {
			wait = time.After(window)
		}
	gather:
		for positions < batchPositions {
			// Once the pass is worth running, what is already queued goes
			// with it and nothing more is waited for.
			if wait != nil && positions >= windowPositions {
				wait = nil
			}
			var j *embedJob
			if wait != nil {
				select {
				case j = <-b.requests:
				case <-wait:
					break gather
				}
			} else {
				select {
				case j = <-b.requests:
				default:
					break gather
				}
			}
			n := 0
			for _, t := range j.texts {
				n += len(t)
			}
			if positions+n > batchPositions {
				held = j
				break gather
			}
			jobs = append(jobs, j)
			positions += n
		}

		var all [][]int32
		for _, j := range jobs {
			all = append(all, j.texts...)
		}
		vecs, err := b.Embedder.Embed(all)
		at := 0
		for _, j := range jobs {
			if err == nil {
				j.vecs = vecs[at : at+len(j.texts)]
			}
			j.err = err
			at += len(j.texts)
			close(j.done)
		}
	}
}

// embedInput is every form "input" takes in the OpenAI API: a string, an
// array of them, an array of identifiers, or an array of arrays of them.
type embedInput struct {
	texts  []string
	tokens [][]int32
}

func (in *embedInput) UnmarshalJSON(raw []byte) error {
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		in.texts = []string{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		in.texts = many
		return nil
	}
	var ids []int32
	if err := json.Unmarshal(raw, &ids); err == nil {
		in.tokens = [][]int32{ids}
		return nil
	}
	var idss [][]int32
	if err := json.Unmarshal(raw, &idss); err == nil {
		in.tokens = idss
		return nil
	}
	return fmt.Errorf("input is neither a string, an array of strings, nor identifiers")
}

func (in *embedInput) count() int { return len(in.texts) + len(in.tokens) }

// prepare turns the input into what a pass reads. A text longer than the
// context is cut, keeping the last identifier — the </s> the model pools with —
// when truncate says so, and refused when it does not.
func (s *Server) prepare(in embedInput, truncate bool) ([][]int32, int, error) {
	limit := s.embed.Context()
	var out [][]int32
	for _, t := range in.texts {
		out = append(out, s.embed.Encode(t))
	}
	for _, ids := range in.tokens {
		out = append(out, append([]int32(nil), ids...))
	}
	total := 0
	for i, ids := range out {
		if len(ids) == 0 {
			return nil, 0, fmt.Errorf("input %d is empty", i)
		}
		if len(ids) > limit {
			if !truncate {
				return nil, 0, fmt.Errorf("input %d is %d tokens, past the context of %d", i, len(ids), limit)
			}
			last := ids[len(ids)-1]
			out[i] = append(ids[:limit-1], last)
		}
		total += len(out[i])
	}
	return out, total, nil
}

// shape applies the requested length — nomic-embed-text-v2 is trained so that
// a prefix of its vector is a vector, which is Matryoshka's bargain — and then
// the unit norm. A prefix is renormalized, as ollama does it.
func shape(v []float32, dims int, normalize bool) []float32 {
	if dims > 0 && dims < len(v) {
		v = v[:dims]
	}
	if normalize {
		nomic.Normalize(v)
	}
	return v
}

type openAIEmbedRequest struct {
	Input          embedInput `json:"input"`
	Model          string     `json:"model"`
	EncodingFormat string     `json:"encoding_format"`
	Dimensions     int        `json:"dimensions"`
}

type openAIEmbedding struct {
	Object    string `json:"object"`
	Index     int    `json:"index"`
	Embedding any    `json:"embedding"`
}

// POST /v1/embeddings
func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req openAIEmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if req.Input.count() == 0 {
		refuse(w, http.StatusBadRequest, "invalid_request_error", "input is empty")
		return
	}
	if req.EncodingFormat != "" && req.EncodingFormat != "float" && req.EncodingFormat != "base64" {
		refuse(w, http.StatusBadRequest, "invalid_request_error", "encoding_format is float or base64")
		return
	}
	// OpenAI refuses a text past the context rather than cutting it, and so
	// does llama-server.
	texts, total, err := s.prepare(req.Input, false)
	if err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	vecs, err := s.embed.Embed(texts)
	if err != nil {
		refuse(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	data := make([]openAIEmbedding, len(vecs))
	for i, v := range vecs {
		v = shape(v, req.Dimensions, true)
		var e any = v
		if req.EncodingFormat == "base64" {
			raw := make([]byte, 4*len(v))
			for j, f := range v {
				binary.LittleEndian.PutUint32(raw[4*j:], math.Float32bits(f))
			}
			e = base64.StdEncoding.EncodeToString(raw)
		}
		data[i] = openAIEmbedding{Object: "embedding", Index: i, Embedding: e}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
		"model":  s.name,
		"usage":  map[string]int{"prompt_tokens": total, "total_tokens": total},
	})
}

// ollamaInput is /api/embed's input: a string or an array of them.
type ollamaEmbedRequest struct {
	Model      string     `json:"model"`
	Input      embedInput `json:"input"`
	Truncate   *bool      `json:"truncate"`
	Dimensions int        `json:"dimensions"`
}

// POST /api/embed
func (s *Server) ollamaEmbed(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req ollamaEmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ollamaRefuse(w, http.StatusBadRequest, err.Error())
		return
	}
	truncate := req.Truncate == nil || *req.Truncate
	texts, total, err := s.prepare(req.Input, truncate)
	if err != nil {
		ollamaRefuse(w, http.StatusBadRequest, err.Error())
		return
	}
	vecs := [][]float32{}
	if len(texts) > 0 {
		if vecs, err = s.embed.Embed(texts); err != nil {
			ollamaRefuse(w, http.StatusInternalServerError, err.Error())
			return
		}
	}
	for i := range vecs {
		vecs[i] = shape(vecs[i], req.Dimensions, true)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"model":             req.Model,
		"embeddings":        vecs,
		"total_duration":    time.Since(start).Nanoseconds(),
		"load_duration":     0,
		"prompt_eval_count": total,
	})
}

type ollamaEmbeddingsRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// POST /api/embeddings, ollama's older endpoint: one prompt, one vector, and
// no normalization, which is how ollama answers it.
func (s *Server) ollamaEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req ollamaEmbeddingsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		ollamaRefuse(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Prompt == "" {
		writeJSON(w, http.StatusOK, map[string]any{"embedding": []float32{}})
		return
	}
	texts, _, err := s.prepare(embedInput{texts: []string{req.Prompt}}, true)
	if err != nil {
		ollamaRefuse(w, http.StatusBadRequest, err.Error())
		return
	}
	vecs, err := s.embed.Embed(texts)
	if err != nil {
		ollamaRefuse(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"embedding": vecs[0]})
}

// ollamaRefuse is ollama's error shape, which is not OpenAI's.
func ollamaRefuse(w http.ResponseWriter, code int, message string) {
	if rec, ok := w.(*recorder); ok {
		rec.reason = message
	}
	writeJSON(w, code, map[string]string{"error": message})
}
