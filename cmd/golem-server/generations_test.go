package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"io"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/krea2"
	"github.com/ThiraSoft/golem/sample"
)

type fakeImager struct{ got krea2.Request }

func (f *fakeImager) WritePNG(w io.Writer, img image.Image, r krea2.Request) error {
	return krea2.WritePNG(w, img, r, "fake")
}

func (f *fakeImager) Generate(r krea2.Request, _ krea2.Progress) (*image.NRGBA, krea2.Timings, error) {
	f.got = r
	return image.NewNRGBA(image.Rect(0, 0, r.Width, r.Height)), krea2.Timings{}, nil
}

func generate(t *testing.T, body string) (*fakeImager, *httptest.ResponseRecorder) {
	t.Helper()
	f := &fakeImager{}
	s := NewServer(nil, nil, "krea2", nil, sample.Params{})
	s.SetImager(f)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(body)))
	return f, rec
}

func TestGenerationsReadsTheFrontsFields(t *testing.T) {
	f, rec := generate(t, `{"prompt":"a cat","negative_prompt":"ugly","size":"512x768","steps":6,"cfg":2.5,"seed":42}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	want := krea2.Request{Prompt: "a cat", Negative: "ugly", Width: 512, Height: 768, Steps: 6, CFG: 2.5, Seed: 42}
	if f.got != want {
		t.Fatalf("drew %+v, want %+v", f.got, want)
	}
	var out struct {
		Data []struct {
			B64  string `json:"b64_json"`
			Seed uint64 `json:"seed"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	raw, err := base64.StdEncoding.DecodeString(out.Data[0].B64)
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if img.Bounds().Dx() != 512 || img.Bounds().Dy() != 768 || out.Data[0].Seed != 42 {
		t.Fatalf("a %v picture, seed %d", img.Bounds(), out.Data[0].Seed)
	}
	if !bytes.Contains(raw, []byte("Seed: 42")) || !bytes.Contains(raw, []byte("golem ")) {
		t.Fatal("the PNG does not say how it was drawn")
	}
}

func TestGenerationsDefaultsAreTheFronts(t *testing.T) {
	f, rec := generate(t, `{"prompt":"a cat"}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if f.got.Width != 768 || f.got.Height != 1024 || f.got.Steps != 8 || f.got.CFG != 1 {
		t.Fatalf("drew %+v", f.got)
	}
	_, rec = generate(t, `{"prompt":"a cat","seed":-1}`)
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
}

func TestGenerationsRefuses(t *testing.T) {
	for _, body := range []string{`{}`, `{"prompt":"x","n":2}`, `{"prompt":"x","size":"big"}`, `{"prompt":"x","response_format":"url"}`, `nope`} {
		if _, rec := generate(t, body); rec.Code != 400 {
			t.Errorf("%s: status %d, want 400", body, rec.Code)
		}
	}
}
