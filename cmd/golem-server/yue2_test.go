package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/sample"
	"github.com/ThiraSoft/golem/yue2"
)

type stubPipeline struct {
	mu        sync.Mutex
	context   int
	generate  func(r yue2.Request, cancel func() bool) (*yue2.Song, error)
	calls     []yue2.Request
	cancelSaw bool
}

func (s *stubPipeline) Context() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.context == 0 {
		return 24576
	}
	return s.context
}

func (s *stubPipeline) Generate(r yue2.Request, cancel func() bool) (*yue2.Song, error) {
	s.mu.Lock()
	s.calls = append(s.calls, r)
	gen := s.generate
	s.mu.Unlock()

	if gen != nil {
		return gen(r, cancel)
	}

	samples := 48000
	audio := make([]float32, 2*samples)
	for i := range audio {
		audio[i] = 0.05
	}
	frames := 25
	latents := make([]float32, frames*64)
	for i := range latents {
		latents[i] = 0.1
	}

	return &yue2.Song{
		Score:   r.ABC,
		Codes:   []int32{10, 20, 30, 40},
		Audio:   audio,
		Samples: samples,
		Latents: latents,
		Frames:  frames,
		LMSeed:  r.LMSeed,
		Seed:    r.Seed,
	}, nil
}

func newTestYuE2Server(stub *stubPipeline) (*Server, *httptest.Server) {
	s := NewServer(nil, nil, "yue2", nil, sample.Params{})
	s.SetYuE2(stub, "YuE2-3B-Q8_0.gguf")
	ts := httptest.NewServer(s.Handler())
	return s, ts
}

func TestYuE2SubmitAndFetchMultipart(t *testing.T) {
	stub := &stubPipeline{}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()

	// 1. Submit a valid song synthesis request
	body := `{"style":"rock","lyrics":"hello world","output_format":"wav16"}`
	resp, err := http.Post(ts.URL+"/synth", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /synth: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /synth returned %d", resp.StatusCode)
	}

	var submitOut struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&submitOut); err != nil {
		t.Fatalf("decode submit response: %v", err)
	}
	if submitOut.ID == "" {
		t.Fatalf("expected non-empty job ID")
	}

	// 2. Poll until done
	deadline := time.Now().Add(5 * time.Second)
	done := false
	for time.Now().Before(deadline) {
		pollResp, err := http.Get(fmt.Sprintf("%s/job?id=%s", ts.URL, submitOut.ID))
		if err != nil {
			t.Fatalf("GET /job: %v", err)
		}
		var st struct {
			Status string `json:"status"`
		}
		err = json.NewDecoder(pollResp.Body).Decode(&st)
		pollResp.Body.Close()
		if err != nil {
			t.Fatalf("decode poll: %v", err)
		}
		if st.Status == "done" {
			done = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !done {
		t.Fatalf("job %s did not complete within deadline", submitOut.ID)
	}

	// 3. Fetch multipart result
	resResp, err := http.Get(fmt.Sprintf("%s/job?id=%s&result=1", ts.URL, submitOut.ID))
	if err != nil {
		t.Fatalf("GET /job result: %v", err)
	}
	defer resResp.Body.Close()
	if resResp.StatusCode != http.StatusOK {
		t.Fatalf("GET result returned status %d", resResp.StatusCode)
	}

	mediaType, params, err := mime.ParseMediaType(resResp.Header.Get("Content-Type"))
	if err != nil {
		t.Fatalf("parse Content-Type: %v", err)
	}
	if mediaType != "multipart/mixed" {
		t.Fatalf("mediaType %s != multipart/mixed", mediaType)
	}
	boundary := params["boundary"]
	if boundary != "yue2-batch-boundary" {
		t.Fatalf("boundary %q != yue2-batch-boundary", boundary)
	}

	rawBytes, err := io.ReadAll(resResp.Body)
	if err != nil {
		t.Fatalf("read result body: %v", err)
	}

	// 4. Parse multipart with standard Go parser
	mr := multipart.NewReader(bytes.NewReader(rawBytes), boundary)

	// Part 1: application/json (replay)
	part1, err := mr.NextPart()
	if err != nil {
		t.Fatalf("next part 1: %v", err)
	}
	if ct := part1.Header.Get("Content-Type"); ct != "application/json" {
		t.Fatalf("part 1 Content-Type %q != application/json", ct)
	}
	p1Bytes, err := io.ReadAll(part1)
	if err != nil {
		t.Fatalf("read part 1: %v", err)
	}
	var replayMap map[string]any
	if err := json.Unmarshal(p1Bytes, &replayMap); err != nil {
		t.Fatalf("unmarshal part 1 JSON: %v", err)
	}
	if replayMap["style"] != "rock" {
		t.Fatalf("replay style %v != rock", replayMap["style"])
	}

	// Part 2: audio/wav
	part2, err := mr.NextPart()
	if err != nil {
		t.Fatalf("next part 2: %v", err)
	}
	if ct := part2.Header.Get("Content-Type"); ct != "audio/wav" {
		t.Fatalf("part 2 Content-Type %q != audio/wav", ct)
	}
	p2Bytes, err := io.ReadAll(part2)
	if err != nil {
		t.Fatalf("read part 2: %v", err)
	}
	if len(p2Bytes) < 44 || string(p2Bytes[:4]) != "RIFF" {
		t.Fatalf("part 2 invalid WAV header")
	}

	// Part 3: application/octet-stream (latents)
	part3, err := mr.NextPart()
	if err != nil {
		t.Fatalf("next part 3: %v", err)
	}
	if ct := part3.Header.Get("Content-Type"); ct != "application/octet-stream" {
		t.Fatalf("part 3 Content-Type %q != application/octet-stream", ct)
	}
	if disp := part3.Header.Get("Content-Disposition"); !strings.Contains(disp, "name=\"latent\"") {
		t.Fatalf("part 3 Content-Disposition %q missing name=latent", disp)
	}
	p3Bytes, err := io.ReadAll(part3)
	if err != nil {
		t.Fatalf("read part 3: %v", err)
	}
	wantLatentLen := 25 * 64 * 4
	if len(p3Bytes) != wantLatentLen {
		t.Fatalf("latents length %d != %d", len(p3Bytes), wantLatentLen)
	}

	// EOF
	_, err = mr.NextPart()
	if err != io.EOF {
		t.Fatalf("expected EOF after part 3, got %v", err)
	}

	// 5. Test parsing with the exact logic used in yue2_worker.py
	workerParts := parseWorkerMultipart(boundary, rawBytes)
	if len(workerParts) != 3 {
		t.Fatalf("worker parser found %d parts, want 3", len(workerParts))
	}
	if workerParts[0].contentType != "application/json" {
		t.Fatalf("worker part 0 ct %q != application/json", workerParts[0].contentType)
	}
	if workerParts[1].contentType != "audio/wav" {
		t.Fatalf("worker part 1 ct %q != audio/wav", workerParts[1].contentType)
	}
	if workerParts[2].contentType != "application/octet-stream" {
		t.Fatalf("worker part 2 ct %q != application/octet-stream", workerParts[2].contentType)
	}
}

type workerPart struct {
	contentType string
	body        []byte
}

func parseWorkerMultipart(boundary string, raw []byte) []workerPart {
	var parts []workerPart
	chunks := bytes.Split(raw, []byte("--"+boundary))
	for _, chunk := range chunks {
		chunk = bytes.Trim(chunk, "\r\n")
		if len(chunk) == 0 || bytes.Equal(chunk, []byte("--")) {
			continue
		}
		idx := bytes.Index(chunk, []byte("\r\n\r\n"))
		if idx == -1 {
			continue
		}
		head := chunk[:idx]
		body := chunk[idx+4:]
		var ct string
		for _, line := range bytes.Split(head, []byte("\r\n")) {
			s := strings.TrimSpace(string(line))
			if strings.HasPrefix(strings.ToLower(s), "content-type:") {
				parts := strings.SplitN(s, ":", 2)
				if len(parts) == 2 {
					ct = strings.TrimSpace(parts[1])
				}
			}
		}
		parts = append(parts, workerPart{contentType: ct, body: body})
	}
	return parts
}

func TestYuE2ResultNotReady404(t *testing.T) {
	unblock := make(chan struct{})
	stub := &stubPipeline{
		generate: func(r yue2.Request, cancel func() bool) (*yue2.Song, error) {
			<-unblock
			samples := 48000
			return &yue2.Song{
				Audio:   make([]float32, 2*samples),
				Samples: samples,
				Latents: make([]float32, 25*64),
				Frames:  25,
			}, nil
		},
	}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()
	defer close(unblock)

	// Submit job
	resp, err := http.Post(ts.URL+"/synth", "application/json", strings.NewReader(`{"output_format":"wav16"}`))
	if err != nil {
		t.Fatalf("POST /synth: %v", err)
	}
	defer resp.Body.Close()

	var out struct {
		ID string `json:"id"`
	}
	json.NewDecoder(resp.Body).Decode(&out)

	// Immediately fetch result: must return 404
	resResp, err := http.Get(fmt.Sprintf("%s/job?id=%s&result=1", ts.URL, out.ID))
	if err != nil {
		t.Fatalf("GET result: %v", err)
	}
	defer resResp.Body.Close()
	if resResp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 before completion, got %d", resResp.StatusCode)
	}
	var errOut struct {
		Error string `json:"error"`
	}
	json.NewDecoder(resResp.Body).Decode(&errOut)
	if errOut.Error != "result not ready" {
		t.Fatalf("expected error 'result not ready', got %q", errOut.Error)
	}
}

func TestYuE2Cancel(t *testing.T) {
	started := make(chan struct{})
	cancelled := make(chan struct{})
	var onceStarted sync.Once
	stub := &stubPipeline{
		generate: func(r yue2.Request, cancel func() bool) (*yue2.Song, error) {
			onceStarted.Do(func() { close(started) })
			for i := 0; i < 100; i++ {
				if cancel != nil && cancel() {
					close(cancelled)
					return nil, yue2.ErrCancelled
				}
				time.Sleep(10 * time.Millisecond)
			}
			return nil, fmt.Errorf("did not see cancel")
		},
	}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/synth", "application/json", strings.NewReader(`{"output_format":"wav16"}`))
	if err != nil {
		t.Fatalf("POST /synth: %v", err)
	}
	defer resp.Body.Close()

	var out struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode /synth response: %v", err)
	}

	// Wait until stub's generate is actively executing
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatalf("pipeline did not start generate")
	}

	// Cancel job
	cancelResp, err := http.Post(fmt.Sprintf("%s/job?id=%s&cancel=1", ts.URL, out.ID), "", nil)
	if err != nil {
		t.Fatalf("POST cancel: %v", err)
	}
	defer cancelResp.Body.Close()
	if cancelResp.StatusCode != http.StatusOK {
		t.Fatalf("cancel returned status %d", cancelResp.StatusCode)
	}
	var cancelOut struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(cancelResp.Body).Decode(&cancelOut); err != nil {
		t.Fatalf("decode cancel response: %v", err)
	}
	if cancelOut.Status != "cancelled" {
		t.Fatalf("cancel response status %q != cancelled", cancelOut.Status)
	}

	select {
	case <-cancelled:
	case <-time.After(2 * time.Second):
		t.Fatalf("pipeline did not receive cancellation")
	}

	// Poll job status: should be cancelled
	pollResp, err := http.Get(fmt.Sprintf("%s/job?id=%s", ts.URL, out.ID))
	if err != nil {
		t.Fatalf("GET /job: %v", err)
	}
	defer pollResp.Body.Close()
	var st struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(pollResp.Body).Decode(&st); err != nil {
		t.Fatalf("decode poll response: %v", err)
	}
	if st.Status != "cancelled" {
		t.Fatalf("final job status %q != cancelled", st.Status)
	}
}

func TestYuE2BadJSON(t *testing.T) {
	stub := &stubPipeline{}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/synth", "application/json", strings.NewReader(`{invalid-json`))
	if err != nil {
		t.Fatalf("POST /synth: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for bad JSON, got %d", resp.StatusCode)
	}
	var errOut struct {
		Error string `json:"error"`
	}
	json.NewDecoder(resp.Body).Decode(&errOut)
	if errOut.Error != "invalid JSON" {
		t.Fatalf("expected 'invalid JSON', got %q", errOut.Error)
	}
}

func TestYuE2MP3Refused(t *testing.T) {
	stub := &stubPipeline{}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/synth", "application/json", strings.NewReader(`{"output_format":"mp3"}`))
	if err != nil {
		t.Fatalf("POST /synth: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for mp3, got %d", resp.StatusCode)
	}
	var errOut struct {
		Error string `json:"error"`
	}
	json.NewDecoder(resp.Body).Decode(&errOut)
	if !strings.Contains(errOut.Error, "MP3") {
		t.Fatalf("expected error mentioning MP3, got %q", errOut.Error)
	}
}

func TestYuE2HealthAndProps(t *testing.T) {
	stub := &stubPipeline{context: 16384}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()

	// GET /health
	hResp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer hResp.Body.Close()
	if hResp.StatusCode != http.StatusOK {
		t.Fatalf("GET /health status %d", hResp.StatusCode)
	}
	var healthOut struct {
		Status string `json:"status"`
	}
	json.NewDecoder(hResp.Body).Decode(&healthOut)
	if healthOut.Status != "ok" {
		t.Fatalf("health status %q != ok", healthOut.Status)
	}

	// GET /props
	pResp, err := http.Get(ts.URL + "/props")
	if err != nil {
		t.Fatalf("GET /props: %v", err)
	}
	defer pResp.Body.Close()
	if pResp.StatusCode != http.StatusOK {
		t.Fatalf("GET /props status %d", pResp.StatusCode)
	}
	var propsOut struct {
		Version    string         `json:"version"`
		Model      string         `json:"model"`
		SampleRate int            `json:"sample_rate"`
		FrameRate  int            `json:"frame_rate"`
		Context    int            `json:"context"`
		Defaults   map[string]any `json:"defaults"`
	}
	if err := json.NewDecoder(pResp.Body).Decode(&propsOut); err != nil {
		t.Fatalf("decode /props: %v", err)
	}
	if propsOut.SampleRate != 48000 {
		t.Fatalf("sample_rate %d != 48000", propsOut.SampleRate)
	}
	if propsOut.FrameRate != 25 {
		t.Fatalf("frame_rate %d != 25", propsOut.FrameRate)
	}
	if propsOut.Context != 16384 {
		t.Fatalf("context %d != 16384", propsOut.Context)
	}
	if propsOut.Model != "YuE2-3B-Q8_0.gguf" {
		t.Fatalf("model %q != YuE2-3B-Q8_0.gguf", propsOut.Model)
	}
	if len(propsOut.Defaults) == 0 {
		t.Fatalf("defaults map is empty")
	}
}

func TestYuE2UnknownJob404(t *testing.T) {
	stub := &stubPipeline{}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()

	r1, err := http.Get(ts.URL + "/job?id=nonexistent")
	if err != nil {
		t.Fatal(err)
	}
	defer r1.Body.Close()
	if r1.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", r1.StatusCode)
	}

	r2, err := http.Get(ts.URL + "/job?id=nonexistent&result=1")
	if err != nil {
		t.Fatal(err)
	}
	defer r2.Body.Close()
	if r2.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", r2.StatusCode)
	}

	r3, err := http.Post(ts.URL+"/job?id=nonexistent&cancel=1", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer r3.Body.Close()
	if r3.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", r3.StatusCode)
	}
}

func TestYuE2Eviction(t *testing.T) {
	stub := &stubPipeline{}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()

	var ids []string
	for i := 0; i < 35; i++ {
		resp, err := http.Post(ts.URL+"/synth", "application/json", strings.NewReader(`{"output_format":"wav16"}`))
		if err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
		var out struct {
			ID string `json:"id"`
		}
		json.NewDecoder(resp.Body).Decode(&out)
		resp.Body.Close()
		ids = append(ids, out.ID)
		// Poll until done before submitting next to make it completed
		for {
			pResp, err := http.Get(fmt.Sprintf("%s/job?id=%s", ts.URL, out.ID))
			if err != nil {
				t.Fatal(err)
			}
			var st struct {
				Status string `json:"status"`
			}
			json.NewDecoder(pResp.Body).Decode(&st)
			pResp.Body.Close()
			if st.Status == "done" {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
	}

	// The oldest jobs (past 32) should have been evicted and return 404
	oldestResp, err := http.Get(fmt.Sprintf("%s/job?id=%s", ts.URL, ids[0]))
	if err != nil {
		t.Fatal(err)
	}
	defer oldestResp.Body.Close()
	if oldestResp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected oldest job %s to be evicted (404), got %d", ids[0], oldestResp.StatusCode)
	}

	// The newest job should still be present
	latestResp, err := http.Get(fmt.Sprintf("%s/job?id=%s", ts.URL, ids[len(ids)-1]))
	if err != nil {
		t.Fatal(err)
	}
	defer latestResp.Body.Close()
	if latestResp.StatusCode != http.StatusOK {
		t.Fatalf("expected latest job %s to be kept (200), got %d", ids[len(ids)-1], latestResp.StatusCode)
	}
}

// TestYuE2DeepQueue verifies that enqueueing more than 64 jobs (> channel buffer)
// does not deadlock the server under mutex contention, and that GET /job continues
// answering promptly for queued jobs.
func TestYuE2DeepQueue(t *testing.T) {
	unblock := make(chan struct{})
	started := make(chan struct{})
	var onceStarted sync.Once
	stub := &stubPipeline{
		generate: func(r yue2.Request, cancel func() bool) (*yue2.Song, error) {
			onceStarted.Do(func() { close(started) })
			<-unblock
			return &yue2.Song{
				Samples: 48000,
				Audio:   make([]float32, 48000*2),
				Latents: make([]float32, 25*64),
			}, nil
		},
	}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()

	// Enqueue 80 jobs (> 64)
	const numJobs = 80
	jobIDs := make([]string, numJobs)
	for i := 0; i < numJobs; i++ {
		resp, err := http.Post(ts.URL+"/synth", "application/json", strings.NewReader(`{"output_format":"wav16"}`))
		if err != nil {
			t.Fatalf("job %d POST /synth failed: %v", i, err)
		}
		var out struct {
			ID string `json:"id"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			resp.Body.Close()
			t.Fatalf("job %d decode error: %v", i, err)
		}
		resp.Body.Close()
		if out.ID == "" {
			t.Fatalf("job %d received empty ID", i)
		}
		jobIDs[i] = out.ID
	}

	// Verify worker picked up and started the first job
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatalf("worker did not start first job within deadline")
	}

	// Verify GET /job answers for queued jobs without deadlock
	for _, idx := range []int{0, 63, 64, numJobs - 1} {
		pollResp, err := http.Get(fmt.Sprintf("%s/job?id=%s", ts.URL, jobIDs[idx]))
		if err != nil {
			t.Fatalf("GET /job for job %d failed: %v", idx, err)
		}
		var st struct {
			Status string `json:"status"`
		}
		err = json.NewDecoder(pollResp.Body).Decode(&st)
		pollResp.Body.Close()
		if err != nil {
			t.Fatalf("job %d decode poll error: %v", idx, err)
		}
		if st.Status != "running" {
			t.Fatalf("job %d expected status 'running', got %q", idx, st.Status)
		}
	}

	// Unblock worker and let jobs complete
	close(unblock)
}

// TestYuE2FailingGenerate verifies that when pipeline generation fails,
// GET /job?id=N returns {"status":"failed","error":"<message>"} and
// GET /job?id=N&result=1 returns HTTP 404.
func TestYuE2FailingGenerate(t *testing.T) {
	stub := &stubPipeline{
		generate: func(r yue2.Request, cancel func() bool) (*yue2.Song, error) {
			return nil, errors.New("simulated VRAM allocation failure")
		},
	}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/synth", "application/json", strings.NewReader(`{"output_format":"wav16"}`))
	if err != nil {
		t.Fatalf("POST /synth: %v", err)
	}
	var out struct {
		ID string `json:"id"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()

	// Poll until status is failed
	var pollOut struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		pollResp, err := http.Get(fmt.Sprintf("%s/job?id=%s", ts.URL, out.ID))
		if err != nil {
			t.Fatalf("GET /job: %v", err)
		}
		json.NewDecoder(pollResp.Body).Decode(&pollOut)
		pollResp.Body.Close()
		if pollOut.Status == "failed" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if pollOut.Status != "failed" {
		t.Fatalf("expected status 'failed', got %q", pollOut.Status)
	}
	if !strings.Contains(pollOut.Error, "simulated VRAM allocation failure") {
		t.Fatalf("expected error message to contain failure details, got %q", pollOut.Error)
	}

	// Fetch result while failed should return 404
	resultResp, err := http.Get(fmt.Sprintf("%s/job?id=%s&result=1", ts.URL, out.ID))
	if err != nil {
		t.Fatalf("GET /job?result=1: %v", err)
	}
	defer resultResp.Body.Close()
	if resultResp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 for failed job result, got %d", resultResp.StatusCode)
	}
}

// TestYuE2WavFormats verifies that wav24 and wav32 outputs are accepted and produced.
func TestYuE2WavFormats(t *testing.T) {
	for _, fmtName := range []string{"wav24", "wav32"} {
		t.Run(fmtName, func(t *testing.T) {
			stub := &stubPipeline{}
			srv, ts := newTestYuE2Server(stub)
			defer srv.Close()
			defer ts.Close()

			resp, err := http.Post(ts.URL+"/synth", "application/json", strings.NewReader(fmt.Sprintf(`{"output_format":%q}`, fmtName)))
			if err != nil {
				t.Fatalf("POST /synth: %v", err)
			}
			var out struct {
				ID string `json:"id"`
			}
			json.NewDecoder(resp.Body).Decode(&out)
			resp.Body.Close()

			// Wait for done
			for i := 0; i < 50; i++ {
				pResp, err := http.Get(fmt.Sprintf("%s/job?id=%s", ts.URL, out.ID))
				if err != nil {
					t.Fatal(err)
				}
				var st struct {
					Status string `json:"status"`
				}
				json.NewDecoder(pResp.Body).Decode(&st)
				pResp.Body.Close()
				if st.Status == "done" {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}

			// Fetch result
			resResp, err := http.Get(fmt.Sprintf("%s/job?id=%s&result=1", ts.URL, out.ID))
			if err != nil {
				t.Fatal(err)
			}
			defer resResp.Body.Close()
			if resResp.StatusCode != http.StatusOK {
				t.Fatalf("expected 200, got %d", resResp.StatusCode)
			}
		})
	}
}

// TestYuE2FIFOOrder verifies that multiple queued jobs are processed in the order received.
func TestYuE2FIFOOrder(t *testing.T) {
	var mu sync.Mutex
	var executed []string

	stub := &stubPipeline{
		generate: func(r yue2.Request, cancel func() bool) (*yue2.Song, error) {
			mu.Lock()
			executed = append(executed, r.Style)
			mu.Unlock()
			return &yue2.Song{
				Samples: 48000,
				Audio:   make([]float32, 48000*2),
				Latents: make([]float32, 25*64),
			}, nil
		},
	}
	srv, ts := newTestYuE2Server(stub)
	defer srv.Close()
	defer ts.Close()

	names := []string{"first", "second", "third", "fourth"}
	var lastID string
	for _, name := range names {
		resp, err := http.Post(ts.URL+"/synth", "application/json", strings.NewReader(fmt.Sprintf(`{"style":%q,"output_format":"wav16"}`, name)))
		if err != nil {
			t.Fatal(err)
		}
		var out struct {
			ID string `json:"id"`
		}
		json.NewDecoder(resp.Body).Decode(&out)
		resp.Body.Close()
		lastID = out.ID
	}

	// Wait until last job is done
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		pResp, err := http.Get(fmt.Sprintf("%s/job?id=%s", ts.URL, lastID))
		if err != nil {
			t.Fatal(err)
		}
		var st struct {
			Status string `json:"status"`
		}
		json.NewDecoder(pResp.Body).Decode(&st)
		pResp.Body.Close()
		if st.Status == "done" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(executed) != len(names) {
		t.Fatalf("expected %d executed jobs, got %d", len(names), len(executed))
	}
	for i, want := range names {
		if executed[i] != want {
			t.Fatalf("job index %d: expected %q, got %q", i, want, executed[i])
		}
	}
}
