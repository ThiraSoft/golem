package engine

import (
	"os"
	"testing"
)

func TestGemmaOffersVisionOnceItHasAProjector(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL")
	proj := os.Getenv("GOLEM_MMPROJ")
	if path == "" || proj == "" {
		t.Skip("set GOLEM_MODEL and GOLEM_MMPROJ to run this test")
	}
	m, err := Open(path, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, ok := m.Media(); ok {
		t.Fatal("a model opened without a projector offers vision")
	}
	if err := m.OpenProjector(proj); err != nil {
		t.Fatal(err)
	}
	v, ok := m.Media()
	if !ok {
		t.Fatal("a model given a projector does not offer vision")
	}
	raw, err := os.ReadFile("../testdata/gemma/shapes.png")
	if err != nil {
		t.Skip("the test image is not on this machine")
	}
	rows, err := v.EncodeImage(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) == 0 {
		t.Fatal("the image became no tokens")
	}
}

func TestQwenRefusesAProjector(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL_QWEN")
	proj := os.Getenv("GOLEM_MMPROJ")
	if path == "" || proj == "" {
		t.Skip("set GOLEM_MODEL_QWEN and GOLEM_MMPROJ to run this test")
	}
	m, err := Open(path, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.OpenProjector(proj); err == nil {
		t.Fatal("a Qwen model accepted a projector")
	}
	if _, ok := m.Media(); ok {
		t.Fatal("a Qwen model offers vision")
	}
}
