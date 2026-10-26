package main

import (
	"reflect"
	"testing"
)

// A vocabulary that wants a BOS gets one at the head of every window, not just
// the first: Gemma without it reads a window as noise, a perplexity of 338
// against 33 on the E2B, and the calibration would measure that noise.
func TestEveryWindowStartsWithBOS(t *testing.T) {
	ids := []int32{10, 11, 12, 13, 14, 15, 16}
	got := cut(ids, []int32{2}, 100, 4)
	want := [][]int32{{2, 10, 11, 12}, {2, 13, 14, 15}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// A prefix follows the BOS on every window, and the text carries on where the
// last window stopped.
func TestPrefixOpensEveryWindow(t *testing.T) {
	ids := []int32{10, 11, 12, 13, 14, 15, 16}
	got := cut(ids, []int32{2, 7}, 100, 5)
	want := [][]int32{{2, 7, 10, 11, 12}, {2, 7, 13, 14, 15}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// One that does not — Qwen's — is cut as it always was.
func TestNoBOSCutsAsBefore(t *testing.T) {
	ids := []int32{10, 11, 12, 13, 14, 15, 16, 17, 18}
	got := cut(ids, nil, 8, 4)
	want := [][]int32{{10, 11, 12, 13}, {14, 15, 16, 17}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}
