package main

// Routes compatible with yue2.cpp's yue-server: POST /synth, GET /job,
// POST /job, GET /health, and GET /props.
//
// Jobs run through a single worker goroutine in FIFO order, keeping the GPU
// bound to one generation at a time. The client polls /job?id=N until done,
// then fetches the multipart/mixed result containing the replay JSON,
// audio WAV, and raw latents.

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/ThiraSoft/golem/internal/version"
	"github.com/ThiraSoft/golem/yue2"
)

// YuE2Pipeline generates songs and reports its context size.
// Both yue2.Pipeline and test stubs satisfy this interface.
type YuE2Pipeline interface {
	Generate(r yue2.Request, cancel func() bool) (*yue2.Song, error)
	Context() int
}

const (
	maxYuE2Jobs           = 32
	yue2MultipartBoundary = "yue2-batch-boundary"
)

type jobStatus string

const (
	statusRunning   jobStatus = "running"
	statusDone      jobStatus = "done"
	statusFailed    jobStatus = "failed"
	statusCancelled jobStatus = "cancelled"
)

type yue2Job struct {
	id         string
	req        yue2.Request
	status     jobStatus
	cancel     atomic.Bool
	resultBody []byte
	mimeType   string
	err        string
	done       chan struct{}
}

type yue2Queue struct {
	pipeline YuE2Pipeline
	model    string

	mu         sync.Mutex
	jobs       map[string]*yue2Job
	order      []string
	workCh     chan *yue2Job
	active     *yue2Job
	stop       chan struct{}
	workerDone chan struct{}
}

func newYuE2Queue(p YuE2Pipeline, model string) *yue2Queue {
	q := &yue2Queue{
		pipeline:   p,
		model:      model,
		jobs:       make(map[string]*yue2Job),
		workCh:     make(chan *yue2Job, 64),
		stop:       make(chan struct{}),
		workerDone: make(chan struct{}),
	}
	go q.worker()
	return q
}

// SetYuE2 lets this server answer song synthesis requests.
func (s *Server) SetYuE2(p YuE2Pipeline, modelName string) {
	s.yue2 = newYuE2Queue(p, modelName)
}

func (q *yue2Queue) Close() {
	select {
	case <-q.stop:
		return
	default:
		close(q.stop)
	}
	<-q.workerDone
}

func (q *yue2Queue) worker() {
	defer close(q.workerDone)
	for {
		select {
		case <-q.stop:
			return
		case job := <-q.workCh:
			q.runJob(job)
		}
	}
}

func (q *yue2Queue) runJob(job *yue2Job) {
	q.mu.Lock()
	if job.cancel.Load() {
		job.status = statusCancelled
		close(job.done)
		q.mu.Unlock()
		return
	}
	q.active = job
	q.mu.Unlock()

	defer func() {
		q.mu.Lock()
		q.active = nil
		close(job.done)
		q.mu.Unlock()
	}()

	cancelFunc := func() bool {
		return job.cancel.Load()
	}

	song, err := q.pipeline.Generate(job.req, cancelFunc)

	q.mu.Lock()
	defer q.mu.Unlock()

	if job.cancel.Load() || errors.Is(err, yue2.ErrCancelled) {
		job.status = statusCancelled
		return
	}
	if err != nil {
		job.status = statusFailed
		job.err = err.Error()
		return
	}

	replayJSON := song.Replay(job.req).JSON()
	audioBytes, err := song.Encode(job.req.OutputFormat, job.req.PeakClip)
	if err != nil {
		job.status = statusFailed
		job.err = err.Error()
		return
	}
	latentBytes := make([]byte, len(song.Latents)*4)
	for i, f := range song.Latents {
		binary.LittleEndian.PutUint32(latentBytes[i*4:], math.Float32bits(f))
	}

	job.resultBody = buildMultipartResponse(replayJSON, audioBytes, latentBytes)
	job.mimeType = "multipart/mixed; boundary=" + yue2MultipartBoundary
	job.status = statusDone
}

func (q *yue2Queue) evictOldestLocked() {
	for len(q.order) > maxYuE2Jobs {
		evicted := false
		for i, id := range q.order {
			job, ok := q.jobs[id]
			if !ok || job.status != statusRunning {
				delete(q.jobs, id)
				q.order = append(q.order[:i], q.order[i+1:]...)
				evicted = true
				break
			}
		}
		if !evicted {
			break
		}
	}
}

func newJobID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic("yue2: crypto/rand failed: " + err.Error())
	}
	return fmt.Sprintf("%016x", binary.LittleEndian.Uint64(b[:]))
}

func (q *yue2Queue) enqueue(req yue2.Request) string {
	q.mu.Lock()
	defer q.mu.Unlock()

	id := newJobID()
	job := &yue2Job{
		id:     id,
		req:    req,
		status: statusRunning,
		done:   make(chan struct{}),
	}
	q.jobs[id] = job
	q.order = append(q.order, id)
	q.evictOldestLocked()
	q.workCh <- job
	return id
}

func (q *yue2Queue) synth(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		yue2Error(w, http.StatusBadRequest, "cannot read request body")
		return
	}
	req, err := validateSynthRequest(body)
	if err != nil {
		yue2Error(w, http.StatusBadRequest, err.Error())
		return
	}
	id := q.enqueue(req)
	writeJSON(w, http.StatusOK, map[string]string{"id": id})
}

func (q *yue2Queue) getJob(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		yue2Error(w, http.StatusNotFound, "job not found")
		return
	}
	q.mu.Lock()
	job, ok := q.jobs[id]
	if !ok {
		q.mu.Unlock()
		yue2Error(w, http.StatusNotFound, "job not found")
		return
	}
	status := job.status
	resultBody := job.resultBody
	mimeType := job.mimeType
	q.mu.Unlock()

	if !r.URL.Query().Has("result") {
		writeJSON(w, http.StatusOK, map[string]string{"status": string(status)})
		return
	}

	if status != statusDone {
		yue2Error(w, http.StatusNotFound, "result not ready")
		return
	}

	w.Header().Set("Content-Type", mimeType)
	w.WriteHeader(http.StatusOK)
	w.Write(resultBody)
}

func (q *yue2Queue) postJob(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		yue2Error(w, http.StatusNotFound, "job not found")
		return
	}
	q.mu.Lock()
	job, ok := q.jobs[id]
	if !ok {
		q.mu.Unlock()
		yue2Error(w, http.StatusNotFound, "job not found")
		return
	}
	if r.URL.Query().Has("cancel") {
		job.cancel.Store(true)
		if job.status == statusRunning {
			job.status = statusCancelled
		}
	}
	status := job.status
	q.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]string{"status": string(status)})
}

func (q *yue2Queue) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (q *yue2Queue) props(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"version":     version.String(),
		"model":       q.model,
		"sample_rate": yue2.SampleRate,
		"frame_rate":  yue2.FrameRate,
		"context":     q.pipeline.Context(),
		"defaults":    defaultsMap(),
	})
}

func yue2Error(w http.ResponseWriter, code int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": message})
}

func validateSynthRequest(body []byte) (yue2.Request, error) {
	req, err := yue2.ParseRequest(body)
	if err != nil {
		return req, fmt.Errorf("invalid JSON")
	}
	if _, err := yue2.ParseCot(req.Cot); err != nil {
		return req, fmt.Errorf("cot must be full, melody or off")
	}
	if req.OutputFormat == yue2.FormatMP3 {
		return req, fmt.Errorf("golem writes no MP3; ask for wav16, wav24 or wav32")
	}
	if req.OutputFormat != yue2.FormatWAV16 && req.OutputFormat != yue2.FormatWAV24 && req.OutputFormat != yue2.FormatWAV32 {
		return req, fmt.Errorf("unknown output format")
	}
	if req.Steps < 1 {
		return req, fmt.Errorf("steps must be positive")
	}
	if req.LMBatchSize < 1 || req.LMBatchSize > 1 || req.SynthBatchSize < 1 || req.SynthBatchSize > 1 {
		return req, fmt.Errorf("batches of 1 song and 1 variation only")
	}
	if err := req.ABCSampling.Valid(); err != nil {
		return req, fmt.Errorf("sampling preset outside the protocol bounds")
	}
	if err := req.SemanticSampling.Valid(); err != nil {
		return req, fmt.Errorf("sampling preset outside the protocol bounds")
	}
	if len(req.Adapters) > 0 {
		return req, fmt.Errorf("adapters are not supported yet")
	}
	if req.SemanticTokens != "" && req.SourceTokens != "" {
		return req, fmt.Errorf("semantic_tokens and source_tokens are exclusive")
	}
	if req.SourceTokens != "" {
		return req, fmt.Errorf("continuations are not supported yet")
	}
	return req.Resolve(), nil
}

func buildMultipartResponse(replayJSON, audioBytes, latentBytes []byte) []byte {
	var buf bytes.Buffer
	buf.WriteString("--" + yue2MultipartBoundary + "\r\nContent-Type: application/json\r\n\r\n")
	buf.Write(replayJSON)
	buf.WriteString("\r\n")

	buf.WriteString("--" + yue2MultipartBoundary + "\r\nContent-Type: audio/wav\r\n\r\n")
	buf.Write(audioBytes)
	buf.WriteString("\r\n")

	buf.WriteString("--" + yue2MultipartBoundary + "\r\nContent-Type: application/octet-stream\r\nContent-Disposition: form-data; name=\"latent\"\r\n\r\n")
	buf.Write(latentBytes)
	buf.WriteString("\r\n")

	buf.WriteString("--" + yue2MultipartBoundary + "--\r\n")
	return buf.Bytes()
}

func defaultsMap() map[string]any {
	d := yue2.DefaultRequest()
	return map[string]any{
		"style":            d.Style,
		"lyrics":           d.Lyrics,
		"abc":              d.ABC,
		"cot":              d.Cot,
		"duration":         d.Duration,
		"lm_seed":          d.LMSeed,
		"seed":             d.Seed,
		"steps":            d.Steps,
		"lm_batch_size":    d.LMBatchSize,
		"synth_batch_size": d.SynthBatchSize,
		"abc_sampling": map[string]any{
			"temperature":        d.ABCSampling.Temperature,
			"top_p":              d.ABCSampling.TopP,
			"top_k":              d.ABCSampling.TopK,
			"repetition_penalty": d.ABCSampling.RepetitionPenalty,
			"penalty_window":     d.ABCSampling.PenaltyWindow,
			"min_tokens":         d.ABCSampling.MinTokens,
			"max_tokens":         d.ABCSampling.MaxTokens,
		},
		"semantic_sampling": map[string]any{
			"temperature":        d.SemanticSampling.Temperature,
			"top_p":              d.SemanticSampling.TopP,
			"top_k":              d.SemanticSampling.TopK,
			"repetition_penalty": d.SemanticSampling.RepetitionPenalty,
			"penalty_window":     d.SemanticSampling.PenaltyWindow,
			"min_tokens":         d.SemanticSampling.MinTokens,
			"max_tokens":         d.SemanticSampling.MaxTokens,
		},
		"semantic_tokens": d.SemanticTokens,
		"cfg_scale":       d.CfgScale,
		"output_format":   d.OutputFormat,
		"peak_clip":       d.PeakClip,
		"mp3_bitrate":     d.MP3Bitrate,
		"source_tokens":   d.SourceTokens,
		"source_start":    d.SourceStart,
		"source_end":      d.SourceEnd,
		"adapters":        []any{},
	}
}
