package flux3

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/krea2"
	"github.com/ThiraSoft/golem/vk"
)

func device(t testing.TB) *vk.Device {
	t.Helper()
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	t.Cleanup(d.Close)
	return d
}

func needFile(t testing.TB, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no %s", path)
	}
}

func TestEncoderMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	needFile(t, krea2.EncoderPath())
	tok := tokenizer(t)

	d := device(t)
	e, err := OpenEncoder(d, krea2.EncoderPath(), false)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	for _, name := range []string{"p0", "p1", "p2"} {
		raw, err := os.ReadFile(filepath.Join(f.dir, "tokens", name+".json"))
		if err != nil {
			t.Fatal(err)
		}
		var rec struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(raw, &rec); err != nil {
			t.Fatal(err)
		}

		ids, real := Tokenize(tok, rec.Text)
		cond, err := e.EncodeMasked(ids, real)
		if err != nil {
			t.Fatal(err)
		}

		want := f.read(t, "encoder/"+name+".cond")
		if len(cond) != len(want) {
			t.Fatalf("%s: %d floats in cond, want %d", name, len(cond), len(want))
		}
		compareRMS(t, name+" cond", cond, want, 0.01)

		if f.has("encoder/" + name + ".hidden_l4") {
			hiddenL4 := f.read(t, "encoder/"+name+".hidden_l4")
			T := len(ids)
			tap0 := make([]float32, T*2560)
			for tIdx := 0; tIdx < T; tIdx++ {
				copy(tap0[tIdx*2560:(tIdx+1)*2560], cond[tIdx*CondWidth:tIdx*CondWidth+2560])
			}
			compareRMS(t, name+" tap0 vs hidden_l4", tap0, hiddenL4, 0.01)
		}
	}
}
