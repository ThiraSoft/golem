package krea2

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"image"
	"image/png"
	"strings"
	"testing"
)

func TestPNGCarriesTheRequest(t *testing.T) {
	r := Request{Prompt: "un renard dans la neige, été", Negative: "flou", Width: 64, Height: 48, Steps: 8, CFG: 1, Seed: 5}
	var buf bytes.Buffer
	if err := WritePNG(&buf, image.NewNRGBA(image.Rect(0, 0, 64, 48)), r, "krea2_turbo_fp8"); err != nil {
		t.Fatal(err)
	}
	// Still a PNG the standard decoder reads, CRCs included.
	img, err := png.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil || img.Bounds().Dx() != 64 {
		t.Fatalf("%v %v", err, img)
	}
	texts := map[string]string{}
	raw := buf.Bytes()[8:]
	for len(raw) >= 12 {
		n := binary.BigEndian.Uint32(raw)
		kind, data := string(raw[4:8]), raw[8:8+n]
		if kind == "iTXt" {
			key, rest, _ := bytes.Cut(data, []byte{0})
			texts[string(key)] = string(rest[4:])
		}
		raw = raw[12+n:]
	}
	if !strings.HasPrefix(texts["Software"], "golem ") {
		t.Errorf("Software = %q", texts["Software"])
	}
	want := "un renard dans la neige, été\nNegative prompt: flou\nSteps: 8, Sampler: er_sde, Schedule type: simple, CFG scale: 1, Seed: 5, Size: 64x48, Model: krea2_turbo_fp8, Version: golem "
	if !strings.HasPrefix(texts["parameters"], want) {
		t.Errorf("parameters = %q", texts["parameters"])
	}
	var back Request
	if err := json.Unmarshal([]byte(texts["golem"]), &back); err != nil || back != r {
		t.Errorf("golem = %q (%v)", texts["golem"], err)
	}
}

func TestParametersNameTheLoRA(t *testing.T) {
	r := Request{Prompt: "a cat", Width: 64, Height: 64, Steps: 8, CFG: 1, Seed: 5, Lora: "style.safetensors", LoraStrength: 0.75}
	if got := r.Parameters("m"); !strings.HasPrefix(got, "a cat <lora:style:0.75>\nSteps: 8,") {
		t.Errorf("parameters = %q", got)
	}
	r.LoraStrength = 0
	if got := r.Parameters("m"); !strings.HasPrefix(got, "a cat\nSteps") {
		t.Errorf("at strength 0, parameters = %q", got)
	}
}
