package main

import (
	"os/exec"
	"strings"
	"testing"
)

// TestFindRecorderPrefersOrder checks the order rather than the outcome: which
// tools are installed is not this test's business, but that pw-record wins over
// arecord when both are there is.
func TestFindRecorderPrefersOrder(t *testing.T) {
	got, err := findRecorderIn(func(name string) (string, error) {
		if name == "arecord" || name == "pw-record" {
			return "/usr/bin/" + name, nil
		}
		return "", exec.ErrNotFound
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != "pw-record" {
		t.Fatalf("got %q, want pw-record", got)
	}
}

// TestFindRecorderNamesAllThree makes the error useful: a machine with none of
// them must be told which three were looked for.
func TestFindRecorderNamesAllThree(t *testing.T) {
	_, err := findRecorderIn(func(string) (string, error) { return "", exec.ErrNotFound })
	if err == nil {
		t.Fatal("want an error")
	}
	for _, name := range []string{"pw-record", "arecord", "ffmpeg"} {
		if !strings.Contains(err.Error(), name) {
			t.Errorf("error does not name %s: %v", name, err)
		}
	}
}
