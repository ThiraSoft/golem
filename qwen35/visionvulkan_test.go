package qwen35

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/imageio"
	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/vk"
)

// openVisionTower binds the projector on its own, without the text model. It
// is what the tower's tests want: the 27B fills this card, and a test that
// opened it too would only ever measure the streaming path.
func openVisionTower(t *testing.T) *VisionTower {
	t.Helper()
	g, err := tensors.OpenGGUF(qwen38mmproj)
	if err != nil {
		t.Skipf("projector: %v", err)
	}
	t.Cleanup(func() { g.Close() })
	cfg, err := LoadVisionConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadVisionWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return NewVisionTower(cfg, w)
}

func visionImage(t *testing.T, path string) *imageio.Image {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("image: %v", err)
	}
	defer f.Close()
	im, err := imageio.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return im
}

// The tower on the card against the tower on the processor, row for row.
//
// It is the coarser of the two tests and the one that says whether the thing
// works at all. TestVulkanVisionTowerMatchesLlamaCpp is what names the block a
// divergence began in.
func TestVulkanVisionTowerMatchesCPU(t *testing.T) {
	heavy.Skip(t, "it puts a model on the card")
	tower := openVisionTower(t)
	im := visionImage(t, filepath.Join("..", "testdata", "gemma", "shapes.png"))

	t0 := time.Now()
	want := tower.Encode(im)
	cpu := time.Since(t0)

	d, err := vk.Open()
	if err != nil {
		t.Skipf("vulkan: %v", err)
	}
	defer d.Close()
	if err := tower.UseVulkan(d); err != nil {
		t.Skipf("upload: %v", err)
	}
	defer tower.Close()

	// Twice: the first pass builds the scratch and the second reuses it, and
	// the second is what an image costs when one has already been encoded.
	got := tower.Encode(im)
	t1 := time.Now()
	got = tower.Encode(im)
	gpu := time.Since(t1)

	t.Logf("resident: %v   processor %v   card %v   (%.0fx)",
		tower.VulkanResident(), cpu.Round(time.Millisecond), gpu.Round(time.Millisecond),
		cpu.Seconds()/gpu.Seconds())

	if len(got) != len(want) {
		t.Fatalf("the card made %d rows and the processor %d", len(got), len(want))
	}
	flatten := func(rows [][]float32) []float32 {
		out := make([]float32, 0, len(rows)*len(rows[0]))
		for _, r := range rows {
			out = append(out, r...)
		}
		return out
	}
	gap, at := worstGap(flatten(got), flatten(want))
	t.Logf("worst gap %.5f of the scale", gap)
	if gap > 0.01 {
		t.Errorf("the two towers disagree by %.4f of the scale, at element %d", gap, at)
	}
}

// The tower on the card against llama.cpp, waypoint by waypoint.
//
// Same fixtures and same bar as TestVisionTowerMatchesLlamaCpp, which holds
// the processor's tower; what this adds is the six waypoints inside a block,
// so that a divergence names not only its block but its stage.
func TestVulkanVisionTowerMatchesLlamaCpp(t *testing.T) {
	heavy.Skip(t, "it puts a model on the card")
	idx, dir := visionFixture(t)
	tower := openVisionTower(t)

	d, err := vk.Open()
	if err != nil {
		t.Skipf("vulkan: %v", err)
	}
	defer d.Close()
	if err := tower.UseVulkan(d); err != nil {
		t.Skipf("upload: %v", err)
	}
	defer tower.Close()

	tower.Trace()
	im := visionImage(t, filepath.Join("..", idx.Image))
	rows := tower.Encode(im)
	if len(rows) != idx.NImageTokens {
		t.Fatalf("%d rows, and llama.cpp made %d — the grid is not the same, so nothing below compares",
			len(rows), idx.NImageTokens)
	}

	const tolerance = 0.02
	names := []string{}
	for _, block := range []int{0, 1, 13, 26} {
		for _, stage := range vk.VisionWaypoints() {
			names = append(names, fmt.Sprintf("%s-%d", stage, block))
		}
	}
	checked := 0
	for _, name := range names {
		meta, ok := idx.Tensors[name]
		if !ok {
			continue
		}
		got := tower.Waypoint(name)
		if got == nil {
			t.Errorf("%s: not traced", name)
			continue
		}
		want := readFixtureFloats(t, filepath.Join(dir, meta.File))
		if len(got) != len(want) {
			t.Errorf("%s: %d values against %d %v", name, len(got), len(want), meta.Ne)
			continue
		}
		gap, at := worstGap(got, want)
		if gap > tolerance {
			t.Errorf("%s: worst gap %.4f of the scale, at element %d (%v against %v)",
				name, gap, at, got[at], want[at])
			continue
		}
		checked++
		t.Logf("%s: worst gap %.5f", name, gap)
	}
	if checked == 0 {
		t.Fatal("no waypoint was recorded, so nothing was compared")
	}
}

// The streaming path against the processor's tower.
//
// Which of the two paths a machine takes depends on what else is on its card,
// so the one it does not take would never be run — and llama.cpp's waypoints
// are checked against the resident form. This forces the other and holds it to
// the same rows.
func TestVulkanVisionTowerStreamed(t *testing.T) {
	heavy.Skip(t, "it puts a model on the card")
	tower := openVisionTower(t)
	im := visionImage(t, filepath.Join("..", "testdata", "gemma", "shapes.png"))
	want := tower.Encode(im)

	t.Setenv("GOLEM_VISION_STREAM", "1")
	d, err := vk.Open()
	if err != nil {
		t.Skipf("vulkan: %v", err)
	}
	defer d.Close()
	if err := tower.UseVulkan(d); err != nil {
		t.Skipf("upload: %v", err)
	}
	defer tower.Close()
	if tower.VulkanResident() {
		t.Fatal("the tower went resident, and this test is about the path that does not")
	}

	got := tower.Encode(im)
	t0 := time.Now()
	got = tower.Encode(im)
	t.Logf("one image, streamed: %v", time.Since(t0).Round(time.Millisecond))

	flat := func(rows [][]float32) []float32 {
		out := make([]float32, 0, len(rows)*len(rows[0]))
		for _, r := range rows {
			out = append(out, r...)
		}
		return out
	}
	if len(got) != len(want) {
		t.Fatalf("%d rows against %d", len(got), len(want))
	}
	gap, at := worstGap(flat(got), flat(want))
	t.Logf("worst gap %.5f of the scale", gap)
	if gap > 0.01 {
		t.Errorf("streamed and on the processor disagree by %.4f of the scale, at element %d", gap, at)
	}
}

// The tower beside the model it belongs to, which is the configuration that
// actually runs.
//
// It asserts only that the tower reaches the card at all through the ordinary
// path — a projector opened, then UseVulkan — and says which side of Prepare's
// choice this machine fell on. Whether the weights fit beside a 27B depends on
// the context the model was opened with, so it is logged and not required.
func TestVulkanVisionTowerBesideTheModel(t *testing.T) {
	heavy.Skip(t, "it puts a model on the card")
	g, err := tensors.OpenGGUF(qwen38)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	m, err := New(g, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.OpenProjector(qwen38mmproj); err != nil {
		t.Skipf("projector: %v", err)
	}
	if err := m.UseVulkan(); err != nil {
		t.Skipf("vulkan: %v", err)
	}
	on, resident := m.VisionVulkan()
	if !on {
		t.Fatal("the tower did not go to the card")
	}
	t.Logf("resident beside the model: %v", resident)

	raw, err := os.ReadFile(filepath.Join("..", "testdata", "gemma", "shapes.png"))
	if err != nil {
		t.Skipf("image: %v", err)
	}
	if _, err := m.EncodeImage(raw); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	rows, err := m.EncodeImage(raw)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("one image: %v for %d rows", time.Since(t0).Round(time.Millisecond), len(rows))
	if len(rows) != 260 {
		t.Fatalf("%d rows, wanted 260", len(rows))
	}
}
