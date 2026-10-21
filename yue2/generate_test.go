package yue2

import (
	"errors"
	"fmt"
	"testing"
)

// TestGenerateTruncated cuts the semantic phase at a budget of three: the
// generation says so, and the cache holds the three tokens and the end token
// the two owed forwards wrote.
func TestGenerateTruncated(t *testing.T) {
	fix := loadFixtures(t)
	ar := sharedAR(t)
	prefix := fix.readInt32(t, "ar/guided_prefix")
	s := SemanticSampling
	s.MinTokens, s.MaxTokens = 3, 3
	g, err := Generate(ar, prefix, nil, 1, s, arSeed, PhaseSemantic, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !g.Truncated || len(g.Tokens) != 3 {
		t.Errorf("truncated %v with %d tokens, want true with 3", g.Truncated, len(g.Tokens))
	}
	if want := len(prefix) + 3 + 1; ar.Pos(0) != want {
		t.Errorf("cache at %d, want %d", ar.Pos(0), want)
	}
}

// TestGenerateForced is the teacher-forced mode: the given tokens, then the
// end token, which is fed once more.
func TestGenerateForced(t *testing.T) {
	fix := loadFixtures(t)
	ar := sharedAR(t)
	prefix := fix.readInt32(t, "ar/guided_prefix")
	forced := fix.readInt32(t, "ar/decode_tokens")[:5]
	s := SemanticSampling
	s.MinTokens, s.MaxTokens = 0, 8
	g, err := Generate(ar, prefix, nil, 1, s, arSeed, PhaseSemantic, forced, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g.Truncated || fmt.Sprint(g.Tokens) != fmt.Sprint(forced) {
		t.Errorf("truncated %v, tokens %v, want false, %v", g.Truncated, g.Tokens, forced)
	}
	if want := len(prefix) + len(forced) + 1; ar.Pos(0) != want {
		t.Errorf("cache at %d, want %d", ar.Pos(0), want)
	}
	stop := 0
	_, err = Generate(ar, prefix, nil, 1, s, arSeed, PhaseSemantic, forced, func() bool { stop++; return stop > 2 })
	if !errors.Is(err, ErrCancelled) {
		t.Errorf("a cancelled generation returned %v, want ErrCancelled", err)
	}
}
