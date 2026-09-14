package main

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/sample"
)

// fakeEmbedder frames a text as one identifier per byte between 0 and 2, and
// answers each text with a vector that says how long it was.
type fakeEmbedder struct {
	context int
	passes  int
}

func (f *fakeEmbedder) Encode(text string) []int32 {
	ids := []int32{0}
	for _, b := range []byte(text) {
		ids = append(ids, int32(b))
	}
	return append(ids, 2)
}

func (f *fakeEmbedder) Context() int { return f.context }

func (f *fakeEmbedder) Embed(texts [][]int32) ([][]float32, error) {
	f.passes++
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = []float32{float32(len(t)), 3, 4, 0}
	}
	return out, nil
}

func embedServer(t *testing.T, ctx int) (*Server, *fakeEmbedder) {
	t.Helper()
	e := &fakeEmbedder{context: ctx}
	s := NewServer(nil, nil, "nomic-test", nil, sample.Params{})
	s.SetEmbedder(e)
	return s, e
}

func postEmbed(t *testing.T, s *Server, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader(body)))
	return w
}

func norm(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

func TestOpenAIEmbeddingsBatchOneReply(t *testing.T) {
	s, e := embedServer(t, 512)
	w := postEmbed(t, s, "/v1/embeddings", `{"model":"x","input":["ab","abcd"]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
	var out struct {
		Object string `json:"object"`
		Data   []struct {
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "list" || len(out.Data) != 2 || out.Data[1].Index != 1 {
		t.Fatalf("%s", w.Body)
	}
	if e.passes != 1 {
		t.Fatalf("two texts took %d passes; one request is one pass", e.passes)
	}
	if out.Usage.PromptTokens != 4+6 {
		t.Fatalf("usage %d", out.Usage.PromptTokens)
	}
	for _, d := range out.Data {
		if math.Abs(norm(d.Embedding)-1) > 1e-6 {
			t.Fatalf("an OpenAI embedding is unit length, got %v", d.Embedding)
		}
	}
}

// The OpenAI client asks for base64 unless told otherwise.
func TestOpenAIEmbeddingsBase64(t *testing.T) {
	s, _ := embedServer(t, 512)
	w := postEmbed(t, s, "/v1/embeddings", `{"input":"ab","encoding_format":"base64"}`)
	var out struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil || len(out.Data) != 1 {
		t.Fatalf("%s", w.Body)
	}
	raw, err := base64.StdEncoding.DecodeString(out.Data[0].Embedding)
	if err != nil || len(raw) != 16 {
		t.Fatalf("%v, %d bytes", err, len(raw))
	}
	first := math.Float32frombits(binary.LittleEndian.Uint32(raw))
	if math.Abs(float64(first)-4/math.Sqrt(16+9+16)) > 1e-6 {
		t.Fatalf("first component %g", first)
	}
}

func TestOpenAIEmbeddingsRefuseWhatDoesNotFit(t *testing.T) {
	s, _ := embedServer(t, 4)
	if w := postEmbed(t, s, "/v1/embeddings", `{"input":"abcdef"}`); w.Code != http.StatusBadRequest {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
	if w := postEmbed(t, s, "/v1/embeddings", `{"input":[]}`); w.Code != http.StatusBadRequest {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
}

func TestOllamaEmbedTruncatesAndNormalizes(t *testing.T) {
	s, _ := embedServer(t, 4)
	w := postEmbed(t, s, "/api/embed", `{"model":"nomic","input":"abcdef","dimensions":2}`)
	if w.Code != http.StatusOK {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
	var out struct {
		Model      string      `json:"model"`
		Embeddings [][]float64 `json:"embeddings"`
		Count      int         `json:"prompt_eval_count"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Model != "nomic" || out.Count != 4 || len(out.Embeddings) != 1 || len(out.Embeddings[0]) != 2 {
		t.Fatalf("%s", w.Body)
	}
	// Cut to four identifiers, the first component is 4; a prefix of two is
	// renormalized.
	if want := 4 / 5.0; math.Abs(out.Embeddings[0][0]-want) > 1e-6 {
		t.Fatalf("%v", out.Embeddings[0])
	}
	if w := postEmbed(t, s, "/api/embed", `{"input":"abcdef","truncate":false}`); w.Code != http.StatusBadRequest {
		t.Fatalf("truncate false let it through: %d", w.Code)
	}
}

// slowEmbedder holds its first pass until told to go, so that requests pile
// up behind it the way they do behind a real pass.
type slowEmbedder struct {
	fakeEmbedder
	entered, gate chan struct{}
	widths        []int
}

func (f *slowEmbedder) Embed(texts [][]int32) ([][]float32, error) {
	if f.passes == 0 {
		close(f.entered)
		<-f.gate
	}
	f.widths = append(f.widths, len(texts))
	return f.fakeEmbedder.Embed(texts)
}

// Requests that arrive while a pass runs are carried together by the next
// one, and every caller gets its own vectors back.
func TestConcurrentRequestsShareAPass(t *testing.T) {
	e := &slowEmbedder{fakeEmbedder: fakeEmbedder{context: 512},
		entered: make(chan struct{}), gate: make(chan struct{})}
	b := newBatcher(e)
	first := make(chan [][]float32)
	go func() {
		v, _ := b.Embed([][]int32{{0, 1, 2}})
		first <- v
	}()
	<-e.entered
	results := make([]chan [][]float32, 5)
	for i := range results {
		results[i] = make(chan [][]float32)
		go func(i int) {
			text := make([]int32, 4+i)
			v, _ := b.Embed([][]int32{text})
			results[i] <- v
		}(i)
	}
	for len(b.requests) < len(results) {
		runtime.Gosched()
	}
	close(e.gate)
	if v := <-first; len(v) != 1 || v[0][0] != 3 {
		t.Fatalf("first request got %v", v)
	}
	for i, r := range results {
		if v := <-r; len(v) != 1 || v[0][0] != float32(4+i) {
			t.Fatalf("request %d got %v", i, v)
		}
	}
	if len(e.widths) != 2 || e.widths[1] != len(results) {
		t.Fatalf("passes carried %v texts; the five that waited should share one", e.widths)
	}
}

func TestOllamaLegacyEmbeddingsIsRaw(t *testing.T) {
	s, _ := embedServer(t, 512)
	w := postEmbed(t, s, "/api/embeddings", `{"model":"nomic","prompt":"ab"}`)
	var out struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Embedding) != 4 || out.Embedding[0] != 4 {
		t.Fatalf("%s", w.Body)
	}
}
