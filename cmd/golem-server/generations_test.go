package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/krea2"
	"github.com/ThiraSoft/golem/sample"
)

type fakeImager struct {
	got            PictureRequest
	size           string
	steps          int
	disallowImages bool
	disallowLoRA   bool
	refuse         error // what Check answers
	drew           bool
}

func (f *fakeImager) Check(r PictureRequest) error { return f.refuse }

func (f *fakeImager) Defaults() (string, int) {
	s, st := f.size, f.steps
	if s == "" {
		s = "768x1024"
	}
	if st == 0 {
		st = 8
	}
	return s, st
}

func (f *fakeImager) SupportsImages() bool { return !f.disallowImages }
func (f *fakeImager) SupportsLoRA() bool   { return !f.disallowLoRA }

func (f *fakeImager) WritePNG(w io.Writer, img image.Image, r PictureRequest) error {
	return krea2.WritePNG(w, img, krea2.Request{
		Prompt: r.Prompt, Negative: r.Negative, Width: r.Width, Height: r.Height,
		Steps: r.Steps, CFG: r.CFG, Seed: r.Seed, HidePrompt: r.HidePrompt,
	}, "fake")
}

func (f *fakeImager) Generate(r PictureRequest, progress krea2.Progress) (*image.NRGBA, krea2.Timings, error) {
	f.got, f.drew = r, true
	if progress != nil {
		for i := 1; i <= r.Steps; i++ {
			progress(i, r.Steps)
		}
	}
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
	want := PictureRequest{Prompt: "a cat", Negative: "ugly", Width: 512, Height: 768, Steps: 6, CFG: 2.5, Seed: 42}
	if f.got.Prompt != want.Prompt || f.got.Negative != want.Negative || f.got.Width != want.Width || f.got.Height != want.Height || f.got.Steps != want.Steps || f.got.CFG != want.CFG || f.got.Seed != want.Seed {
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

func TestGenerationsCanHideThePrompt(t *testing.T) {
	_, rec := generate(t, `{"prompt":"un secret","negative_prompt":"un autre","seed":3,"hide_prompt":true}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var out struct {
		Data []struct {
			B64 string `json:"b64_json"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	raw, _ := base64.StdEncoding.DecodeString(out.Data[0].B64)
	if bytes.Contains(raw, []byte("secret")) || bytes.Contains(raw, []byte("autre")) || !bytes.Contains(raw, []byte("Seed: 3")) {
		t.Fatal("the PNG carries the prompt, or not the settings")
	}
}

func TestGenerationsReadsTheLoRA(t *testing.T) {
	f, rec := generate(t, `{"prompt":"a cat","seed":1,"lora":"style.safetensors","lora_strength":0.6}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if f.got.Lora != "style.safetensors" || f.got.LoraStrength != 0.6 {
		t.Fatalf("drew %+v", f.got)
	}
	f, _ = generate(t, `{"prompt":"a cat","lora":"style.safetensors"}`)
	if f.got.LoraStrength != 1 {
		t.Fatalf("with no strength, drew %+v", f.got)
	}
	if _, rec = generate(t, `{"prompt":"a cat","lora":"x.safetensors","lora_strength":2.5}`); rec.Code != 400 {
		t.Fatalf("a strength of 2.5: status %d", rec.Code)
	}
	if _, rec = generate(t, `{"prompt":"a cat","lora":"../unet/krea2_turbo_fp8.safetensors"}`); rec.Code != 400 {
		t.Fatalf("a path for a LoRA: status %d", rec.Code)
	}
}

func TestGenerationsStreamsTheSteps(t *testing.T) {
	_, rec := generate(t, `{"prompt":"a cat","steps":3,"seed":9,"stream":true}`)
	if rec.Code != 200 || rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status %d, %s", rec.Code, rec.Header().Get("Content-Type"))
	}
	var kinds []string
	var last map[string]any
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		if data, ok := strings.CutPrefix(line, "data: "); ok {
			last = map[string]any{}
			if err := json.Unmarshal([]byte(data), &last); err != nil {
				t.Fatal(err)
			}
			kinds = append(kinds, fmt.Sprintf("%v %v", last["type"], last["step"]))
		}
	}
	want := "[image_generation.progress 1 image_generation.progress 2 image_generation.progress 3 image_generation.completed <nil>]"
	if fmt.Sprint(kinds) != want {
		t.Fatalf("events %v", kinds)
	}
	raw, err := base64.StdEncoding.DecodeString(last["b64_json"].(string))
	if err != nil || !bytes.Contains(raw, []byte("Seed: 9")) || last["seed"].(float64) != 9 {
		t.Fatalf("the last event's picture: %v", err)
	}
}

func TestGenerationsDecodesBase64Images(t *testing.T) {
	// Create a tiny 2x2 test image encoded as PNG
	tiny := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	var buf bytes.Buffer
	if err := png.Encode(&buf, tiny); err != nil {
		t.Fatal(err)
	}
	b64Raw := base64.StdEncoding.EncodeToString(buf.Bytes())
	b64DataURI := "data:image/png;base64," + b64Raw

	body := fmt.Sprintf(`{"prompt":"a cat","images":["%s","%s"]}`, b64Raw, b64DataURI)
	f, rec := generate(t, body)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if len(f.got.Images) != 2 {
		t.Fatalf("got %d images, want 2", len(f.got.Images))
	}
	for i, img := range f.got.Images {
		if img.Bounds().Dx() != 2 || img.Bounds().Dy() != 2 {
			t.Errorf("image %d bounds %v, want 2x2", i, img.Bounds())
		}
	}

	// Bad base64 image rejected with 400
	if _, rec := generate(t, `{"prompt":"a cat","images":["not_valid_base64!!!"]}`); rec.Code != 400 {
		t.Errorf("status %d, want 400", rec.Code)
	}
}

func TestGenerationsRefusesImagesOnKrea2(t *testing.T) {
	tiny := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	var buf bytes.Buffer
	if err := png.Encode(&buf, tiny); err != nil {
		t.Fatal(err)
	}
	b64Raw := base64.StdEncoding.EncodeToString(buf.Bytes())

	f := &fakeImager{disallowImages: true}
	s := NewServer(nil, nil, "krea2", nil, sample.Params{})
	s.SetImager(f)
	rec := httptest.NewRecorder()
	body := fmt.Sprintf(`{"prompt":"a cat","images":["%s"]}`, b64Raw)
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(body)))
	if rec.Code != 400 {
		t.Fatalf("status %d, want 400", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "images") {
		t.Errorf("error %s does not name 'images'", rec.Body.String())
	}
}

func TestGenerationsRefusesLoRAOnFlux3(t *testing.T) {
	f := &fakeImager{disallowLoRA: true, size: "1024x1024", steps: 16}
	s := NewServer(nil, nil, "flux3", nil, sample.Params{})
	s.SetImager(f)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"prompt":"a cat","lora":"style.safetensors"}`)))
	if rec.Code != 400 {
		t.Fatalf("status %d, want 400", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "lora") {
		t.Errorf("error %s does not name 'lora'", rec.Body.String())
	}
}

func TestGenerationsCapsImagesCount(t *testing.T) {
	tiny := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	var buf bytes.Buffer
	if err := png.Encode(&buf, tiny); err != nil {
		t.Fatal(err)
	}
	b64Raw := base64.StdEncoding.EncodeToString(buf.Bytes())

	// 5 images exceeds cap of 4
	body := fmt.Sprintf(`{"prompt":"a cat","images":["%s","%s","%s","%s","%s"]}`, b64Raw, b64Raw, b64Raw, b64Raw, b64Raw)
	_, rec := generate(t, body)
	if rec.Code != 400 {
		t.Fatalf("status %d, want 400", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "images") {
		t.Errorf("error %s does not name 'images'", rec.Body.String())
	}
}

func TestGenerationsRefusesOversizedImageDimensions(t *testing.T) {
	// Create an image with a side past 8192 (8193 x 1)
	big := image.NewNRGBA(image.Rect(0, 0, 8193, 1))
	var buf bytes.Buffer
	if err := png.Encode(&buf, big); err != nil {
		t.Fatal(err)
	}
	b64Raw := base64.StdEncoding.EncodeToString(buf.Bytes())

	body := fmt.Sprintf(`{"prompt":"a cat","images":["%s"]}`, b64Raw)
	_, rec := generate(t, body)
	if rec.Code != 400 {
		t.Fatalf("status %d, want 400", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "limit") && !strings.Contains(rec.Body.String(), "8192") {
		t.Errorf("error %s does not name limit", rec.Body.String())
	}
}

func TestGenerationsFlux3Defaults(t *testing.T) {
	f := &fakeImager{size: "2048x2048", steps: 16, disallowLoRA: true}
	s := NewServer(nil, nil, "flux3", nil, sample.Params{})
	s.SetImager(f)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(`{"prompt":"a cat"}`)))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if f.got.Width != 2048 || f.got.Height != 2048 || f.got.Steps != 16 {
		t.Fatalf("drew %+v, want 2048x2048, 16 steps", f.got)
	}
}

func tinyPNG(t *testing.T) string {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewNRGBA(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func TestFlux3MatchImageCarried(t *testing.T) {
	f := &fakeImager{size: "2048x2048", steps: 16, disallowLoRA: true}
	s := NewServer(nil, nil, "flux3", nil, sample.Params{})
	s.SetImager(f)
	rec := httptest.NewRecorder()
	body := fmt.Sprintf(`{"prompt":"a cat","match_image":true,"images":["%s"]}`, tinyPNG(t))
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if !f.got.MatchImage {
		t.Error("match_image did not reach the imager")
	}
	if toFlux3(f.got).MatchImage != true {
		t.Error("toFlux3 drops MatchImage")
	}
}

func TestFlux3MatchImageWithoutImagesIs400(t *testing.T) {
	_, rec := generate(t, `{"prompt":"a cat","match_image":true}`)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "match_image") {
		t.Fatalf("status %d %s, want 400 naming match_image", rec.Code, rec.Body)
	}
}

func TestFlux3DefaultSize(t *testing.T) {
	if size, steps := (&flux3Imager{}).Defaults(); size != "2048x2048" || steps != 16 {
		t.Errorf("FLUX 3 defaults %s, %d steps; want 2048x2048, 16", size, steps)
	}
}

func TestFlux3SizePastLimitIs400(t *testing.T) {
	f := &fakeImager{size: "2048x2048", steps: 16, disallowLoRA: true,
		refuse: fmt.Errorf("flux3: 2560 × 2560 is past the 4456448 pixels this pipeline was opened for")}
	s := NewServer(nil, nil, "flux3", nil, sample.Params{})
	s.SetImager(f)
	for _, stream := range []bool{false, true} {
		rec := httptest.NewRecorder()
		body := fmt.Sprintf(`{"prompt":"a cat","size":"2560x2560","stream":%v}`, stream)
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(body)))
		if rec.Code != 400 || !strings.Contains(rec.Body.String(), "4456448") {
			t.Errorf("stream %v: status %d %s, want 400 naming 4456448", stream, rec.Code, rec.Body)
		}
	}
	if f.drew {
		t.Error("the imager drew a picture its Check refused")
	}
}

// TestFlux3TakesAPhonePhoto: a 13 MP phone photo (4160 x 3120) is what
// match_image is for; it reaches the imager.
func TestFlux3TakesAPhonePhoto(t *testing.T) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewNRGBA(image.Rect(0, 0, 4160, 3120))); err != nil {
		t.Fatal(err)
	}
	f := &fakeImager{size: "2048x2048", steps: 1, disallowLoRA: true}
	s := NewServer(nil, nil, "flux3", nil, sample.Params{})
	s.SetImager(f)
	rec := httptest.NewRecorder()
	body := fmt.Sprintf(`{"prompt":"a cat","match_image":true,"images":["%s"]}`, base64.StdEncoding.EncodeToString(buf.Bytes()))
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if b := f.got.Images[0].Bounds(); b.Dx() != 4160 || b.Dy() != 3120 {
		t.Errorf("the imager got %v", b)
	}
}
