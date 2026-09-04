package stt

import (
	"context"
	"strings"
	"sync"
	"testing"
)

// TestSeparateStreamsKeepTheirOwnScratch is the guard on a Model being shared.
//
// A Model is opened once and read by every transcription at once. That was only
// true of its weights: the Mimi encoder's layers carried the scratch of one pass
// on themselves, so two streams wrote each other's projections and each read a
// mixture. Nothing failed — a transcript simply came back partly of the other
// sound, which is the kind of fault a word error rate would have called a bad
// day. This is two ordinary streams, no group and no card, and they must each
// say exactly what one stream alone says.
func TestSeparateStreamsKeepTheirOwnScratch(t *testing.T) {
	m := testModel(t)
	clip := speech(t)
	want := transcribeAlone(t, m, clip)

	const n = 2
	got := make([]string, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			live := m.Stream(context.Background())
			var text strings.Builder
			done := make(chan struct{})
			go func() {
				defer close(done)
				for seg := range live.Text() {
					text.WriteString(seg.Text)
				}
			}()
			live.Write(clip)
			live.Close()
			<-done
			got[i] = strings.TrimSpace(text.String())
		}(i)
	}
	wg.Wait()
	for i, g := range got {
		if g != want {
			t.Errorf("stream %d:\n got %q\nwant %q", i, g, want)
		}
	}
}
