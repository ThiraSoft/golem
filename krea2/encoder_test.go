package krea2

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

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

// The conditioning of each recorded prompt. ComfyUI runs the text model in
// fp16 and this in float32 with fp16 products, so they part by the size of
// fp16's rounding carried through thirty-five layers.
func TestEncoderMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	needFile(t, EncoderPath())
	v := vocab(t)
	e, err := OpenEncoder(device(t), EncoderPath())
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	for _, name := range []string{"short", "portrait", "weighted"} {
		raw, err := os.ReadFile(filepath.Join(f.dir, "tokens", name+".json"))
		if err != nil {
			t.Fatal(err)
		}
		var rec struct{ Text string }
		json.Unmarshal(raw, &rec)
		p := Tokenize(v, rec.Text)
		cond, seq, err := e.Encode(p)
		if err != nil {
			t.Fatal(err)
		}
		want := f.read(t, "encoder/"+name+"/cond")
		if seq*CondWidth != len(want) {
			t.Fatalf("%s: %d tokens of conditioning, want %d", name, seq, len(want)/CondWidth)
		}
		for j := range Taps {
			got := make([]float32, 0, seq*encWidth)
			exp := make([]float32, 0, seq*encWidth)
			for s := 0; s < seq; s++ {
				got = append(got, cond[s*CondWidth+j*encWidth:s*CondWidth+(j+1)*encWidth]...)
				exp = append(exp, want[s*CondWidth+j*encWidth:s*CondWidth+(j+1)*encWidth]...)
			}
			compare(t, name+" tap "+itoa(Taps[j]), got, exp, 5e-2)
		}
	}
}

func itoa(i int) string { return string(rune('0'+i/10)) + string(rune('0'+i%10)) }
