package gemma

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/imageio"
	"github.com/ThiraSoft/golem/tensors"
)

// One picture through the whole tower, which is what a multimodal prompt pays
// on top of reading its text. It is paid once per picture rather than once per
// token, which is why nothing in the tower is repacked or quantized.
func BenchmarkVisionEncode(b *testing.B) {
	if os.Getenv("GOLEM_MMPROJ") == "" {
		b.Skip("set GOLEM_MMPROJ to run this benchmark")
	}
	benchmarkEncode(b, openMMProj(b))
}

// And the 12B's projector, which is not a tower: one product against a
// hundred-megabyte weight, and three norms. What it costs is what reading that
// weight costs.
func BenchmarkUnifiedVisionEncode(b *testing.B) {
	if os.Getenv("GOLEM_MMPROJ_12B") == "" {
		b.Skip("set GOLEM_MMPROJ_12B to run this benchmark")
	}
	benchmarkEncode(b, openMMProj12(b))
}

func benchmarkEncode(b *testing.B, g *tensors.GGUF) {
	cfg, err := LoadVisionConfig(g)
	if err != nil {
		b.Fatal(err)
	}
	w, err := LoadVisionWeights(g, cfg)
	if err != nil {
		b.Fatal(err)
	}
	tower := NewVisionTower(cfg, w)

	f, err := os.Open("../testdata/gemma/shapes.png")
	if err != nil {
		b.Skip("the test image is not on this machine")
	}
	defer f.Close()
	im, err := imageio.Decode(f)
	if err != nil {
		b.Fatal(err)
	}

	tokens := len(tower.Encode(im))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tower.Encode(im)
	}
	b.StopTimer()
	b.ReportMetric(float64(tokens), "tokens/image")
}
