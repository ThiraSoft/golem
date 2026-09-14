package nomic

import (
	"context"
	"errors"
	"testing"
	"time"
)

// A caller waiting behind another pass stops waiting when its context ends,
// and the turn is still the other caller's to give back.
func TestEmbedContextGivesUpWaiting(t *testing.T) {
	m := &Model{turn: make(chan struct{}, 1)}
	m.turn <- struct{}{} // another pass holds the model

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := m.EmbedContext(ctx, [][]int32{{0, 1, 2}}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v, want the deadline", err)
	}
	if len(m.turn) != 1 {
		t.Fatal("the waiting caller took or released a turn it never had")
	}
}

// A context already over runs no pass, and the model is free afterwards.
func TestEmbedContextCancelled(t *testing.T) {
	m := openModel(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.EmbedContext(ctx, [][]int32{m.Tokenize("hello")}); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if _, err := m.Embed([][]int32{m.Tokenize("hello")}); err != nil {
		t.Fatalf("the model stayed taken: %v", err)
	}
}
