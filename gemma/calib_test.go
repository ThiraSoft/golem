package gemma

import (
	"fmt"
	"testing"
)

// Every matrix of a dense block reads one of four activations, and the head a
// fifth. The tap has to name each of them, once a block, at the width of the
// matrix that reads it — a site missing or filed under the wrong width is a
// matrix the converter quantizes blind, which nothing downstream notices.
func TestCalibSeesEverySite(t *testing.T) {
	m := openEngine(t, 64)
	type seen struct{ rows, cols int }
	got := map[string]seen{}
	Calib = func(block int, site string, rows [][]float32) {
		key := fmt.Sprintf("%d/%s", block, site)
		s := got[key]
		s.rows += len(rows)
		s.cols = len(rows[0])
		got[key] = s
	}
	defer func() { Calib = nil }()

	tokens := []int32{2, 818, 5279, 529, 7001, 563, 3285, 236761}
	m.ForwardBatch(tokens, 0)

	cfg := m.Cfg
	want := map[string]int{"-1/head": cfg.Dim}
	for _, bc := range cfg.Blocks {
		want[fmt.Sprintf("%d/qkv", bc.Index)] = cfg.Dim
		want[fmt.Sprintf("%d/o", bc.Index)] = bc.Heads * bc.HeadDim
		want[fmt.Sprintf("%d/gateup", bc.Index)] = cfg.Dim
		want[fmt.Sprintf("%d/down", bc.Index)] = bc.FFN
	}
	for key, cols := range want {
		s, ok := got[key]
		if !ok {
			t.Errorf("site %s never tapped", key)
			continue
		}
		if s.cols != cols || s.rows != len(tokens) {
			t.Errorf("site %s: %d rows of %d, want %d rows of %d", key, s.rows, s.cols, len(tokens), cols)
		}
	}
	for key := range got {
		if _, ok := want[key]; !ok {
			t.Errorf("site %s tapped but no matrix reads it", key)
		}
	}
}
