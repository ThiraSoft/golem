package flux3

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// vramUsed is what the card holds, summed over its sysfs nodes; false where
// the driver does not say.
func vramUsed() (int64, bool) {
	paths, _ := filepath.Glob("/sys/class/drm/card*/device/mem_info_vram_used")
	var total int64
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
		if err != nil {
			continue
		}
		total += n
	}
	return total, len(paths) > 0
}

// stageRequest is what dump.py writes in a stage's request.json.
type stageRequest struct {
	Prompt    string   `json:"prompt"`
	Seed      uint64   `json:"seed"`
	Steps     int      `json:"steps"`
	Width     int      `json:"width"`
	Height    int      `json:"height"`
	RefShapes [][2]int `json:"ref_shapes"`
}

func readStageRequest(t testing.TB, f *fixtures, stage string) stageRequest {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(f.dir, stage, "request.json"))
	if err != nil {
		t.Skipf("%s not recorded: %v", stage, err)
	}
	var r stageRequest
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

// refPlanes splits ref_latents (tokens of 128 channels, references one after
// the other) back into the planes SetRefs takes.
func refPlanes(tokens []float32, shapes [][2]int) [][]float32 {
	var out [][]float32
	off := 0
	for _, sh := range shapes {
		hw := sh[0] * sh[1]
		p := make([]float32, hw*ditInChannels)
		for tok := 0; tok < hw; tok++ {
			for ch := 0; ch < ditInChannels; ch++ {
				p[ch*hw+tok] = tokens[(off+tok)*ditInChannels+ch]
			}
		}
		out = append(out, p)
		off += hw
	}
	return out
}

func TestDiTArenaFloats(t *testing.T) {
	one := ditArenaFloats(MaxImageTokens, MaxRefTokens) * 4
	four := ditArenaFloats(17408, MaxRefs*RefTokens) * 4
	t.Logf("arena: 1 MP + 1 ref %d bytes, 4.25 MP + 4 refs %d bytes", one, four)
	if one >= 2_000_000_000 {
		t.Errorf("1 MP arena %d bytes, want under the 2.0 GB it took before the MLP went in chunks", one)
	}
	if uint64(four) >= 1<<32 {
		t.Errorf("4.25 MP + 4 refs arena %d bytes, want under 4 GiB", four)
	}
	if uint64(ditArenaFloats(40000, 16384))*4 < 1<<32 {
		t.Error("an arena for 40000 image tokens fits in 4 GiB, want past it")
	}
}

func TestOpenDiTSizedRefusesPast4GiB(t *testing.T) {
	path := ditPath()
	needFile(t, path)
	d := device(t)
	start := time.Now()
	m, err := OpenDiTSized(d, path, PrecisionFP16, 40000, 16384)
	if err == nil {
		m.Close()
		t.Fatal("OpenDiTSized for 40000 image tokens succeeded, want an error")
	}
	if took := time.Since(start); took > 5*time.Second {
		t.Errorf("refusing took %v, want it before any weight is read", took)
	}
	if !strings.Contains(err.Error(), "40000") || !strings.Contains(err.Error(), "4 GiB") {
		t.Errorf("error %q does not name 40000 tokens and the 4 GiB limit", err)
	}
}

func TestDiT4MPMatchesComfyUI(t *testing.T) {
	path := ditPath()
	needFile(t, path)
	f := loadFixtures(t)
	d := device(t)

	m, err := OpenDiTSized(d, path, PrecisionFP16, 4456448/256, MaxRefs*RefTokens)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	t.Logf("FP16 weights %d MB, arena %d MB", m.WeightBytes()>>20, m.ArenaBytes()>>20)

	for _, stage := range []string{"t2i4", "mi2i4"} {
		req := readStageRequest(t, f, stage)
		inX := f.read(t, stage+"/in.x")
		inT := f.read(t, stage+"/in.t")[0]
		inNext := f.read(t, stage+"/in.next")[0]
		inCtx := f.read(t, stage+"/in.ctx")
		want := f.read(t, stage+"/out")
		if err := m.SetText(0, inCtx, len(inCtx)/CondWidth); err != nil {
			t.Fatal(err)
		}
		var refs [][]float32
		if len(req.RefShapes) > 0 {
			refs = refPlanes(f.read(t, stage+"/ref_latents"), req.RefShapes)
		}
		if err := m.SetRefs(0, refs, req.RefShapes); err != nil {
			t.Fatal(err)
		}
		h, w := req.Height/16, req.Width/16
		got, err := m.Step(0, inX, h, w, inT, inNext)
		if err != nil {
			t.Fatalf("%s: %v", stage, err)
		}
		compareRMS(t, stage+"/out", got, want, 0.03)

		start := time.Now()
		if _, err := m.Step(0, inX, h, w, inT, inNext); err != nil {
			t.Fatal(err)
		}
		took := time.Since(start)
		used, ok := vramUsed()
		t.Logf("%s %dx%d, %d ref tokens, %d text tokens: step %v, VRAM used %d MB (known %v)",
			stage, req.Width, req.Height, m.refTokens[0], m.ctxTokens[0], took, used>>20, ok)
	}
}
