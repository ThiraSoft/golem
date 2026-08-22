package gemma

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/imageio"
)

// One picture through the whole tower, which is what a multimodal prompt pays
// on top of reading its text. It is paid once per picture rather than once per
// token, which is why nothing in the tower is repacked or quantized.
func BenchmarkVisionEncode(b *testing.B) {
	if os.Getenv("GOLEM_MMPROJ") == "" {
		b.Skip("set GOLEM_MMPROJ to run this benchmark")
	}
	g := openMMProj(b)
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
