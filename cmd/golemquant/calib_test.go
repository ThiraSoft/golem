package main

import (
	"os"
	"strings"
	"testing"
)

// A Gemma checkpoint is calibrated rather than skipped: the salience is worth
// twenty points of perplexity, and a conversion without it is not one to ship.
// Every block's four sites and the head come back, at the widths the
// converter will look them up by.
func TestGemmaIsCalibrated(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("set GOLEM_MODEL to a Gemma 4 GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("GOLEM_MODEL names %s, which is not there", path)
	}
	sal, _ := calibrate(path, strings.Repeat(calibText+" ", 8), 128, 64, 0, 64, 0, false)
	if len(sal) == 0 {
		t.Fatal("nothing calibrated")
	}
	for _, site := range []string{"qkv", "o", "gateup", "down"} {
		if _, ok := sal["0/"+site]; !ok {
			t.Errorf("block 0 has no %s site", site)
		}
	}
	if _, ok := sal[headKey]; !ok {
		t.Errorf("the head has no site")
	}
}
