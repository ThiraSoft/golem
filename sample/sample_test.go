package sample

import (
	"math"
	"testing"
)

// A distribution that is easy to reason about: the probabilities are 0.5, 0.3,
// 0.15 and 0.05 to within a rounding error, so every threshold below falls
// somewhere obvious.
func logits() []float32 {
	p := []float64{0.5, 0.3, 0.15, 0.05}
	out := make([]float32, len(p))
	for i, v := range p {
		out[i] = float32(math.Log(v))
	}
	return out
}

func TestGreedyTakesTheHighestAndTiesGoLow(t *testing.T) {
	if got := Greedy([]float32{1, 3, 2}); got != 1 {
		t.Fatalf("greedy chose %d", got)
	}
	if got := Greedy([]float32{2, 2, 1}); got != 0 {
		t.Fatalf("a tie chose %d, and should go to the lower identifier", got)
	}
}

func TestSoftmaxSumsToOne(t *testing.T) {
	p := Softmax(logits())
	var sum float32
	for _, v := range p {
		sum += v
	}
	if math.Abs(float64(sum)-1) > 1e-5 {
		t.Fatalf("the probabilities sum to %v", sum)
	}
	if math.Abs(float64(p[0])-0.5) > 1e-5 {
		t.Fatalf("the first probability is %v, expected 0.5", p[0])
	}
}

// Softmax has to survive the size of a real logit row, where the raw
// exponential would overflow.
func TestSoftmaxIsShiftInvariant(t *testing.T) {
	small := Softmax([]float32{1, 2, 3})
	large := Softmax([]float32{801, 802, 803})
	for i := range small {
		if math.Abs(float64(small[i]-large[i])) > 1e-6 {
			t.Fatalf("shifting the logits changed the distribution: %v against %v", small, large)
		}
	}
}

// A temperature of zero is the greedy choice, whatever the other parameters say.
func TestZeroTemperatureIsGreedy(t *testing.T) {
	s := New(Params{Temperature: 0, TopK: 64, TopP: 0.95, Seed: 7})
	for i := 0; i < 8; i++ {
		if got := s.Pick(logits()); got != 0 {
			t.Fatalf("draw %d gave %d", i, got)
		}
	}
}

func TestTopKOfOneLeavesNothingToChoose(t *testing.T) {
	s := New(Params{Temperature: 1, TopK: 1, TopP: 1, Seed: 3})
	for i := 0; i < 32; i++ {
		if got := s.Pick(logits()); got != 0 {
			t.Fatalf("draw %d gave %d with a single candidate", i, got)
		}
	}
}

// 0.5 does not reach 0.7, 0.5+0.3 does: the token that crosses the threshold is
// kept and the tail is cut.
func TestTopPKeepsTheTokenThatCrossesIt(t *testing.T) {
	s := New(Params{Temperature: 1, TopK: 0, TopP: 0.7, Seed: 11})
	seen := map[int32]int{}
	for i := 0; i < 4000; i++ {
		seen[s.Pick(logits())]++
	}
	if seen[2] != 0 || seen[3] != 0 {
		t.Fatalf("the tail was drawn: %v", seen)
	}
	if seen[0] == 0 || seen[1] == 0 {
		t.Fatalf("one of the two kept tokens was never drawn: %v", seen)
	}
	// Renormalised, they stand at 0.625 and 0.375.
	share := float64(seen[0]) / 4000
	if math.Abs(share-0.625) > 0.03 {
		t.Fatalf("the first token took %v of the draws, expected 0.625", share)
	}
}

// A threshold below the first probability still leaves one candidate: an empty
// distribution has nothing to draw from.
func TestTopPAlwaysKeepsOne(t *testing.T) {
	s := New(Params{Temperature: 1, TopK: 0, TopP: 0.01, Seed: 5})
	for i := 0; i < 16; i++ {
		if got := s.Pick(logits()); got != 0 {
			t.Fatalf("draw %d gave %d", i, got)
		}
	}
}

func TestUntouchedParametersReproduceTheDistribution(t *testing.T) {
	s := New(Params{Temperature: 1, TopK: 0, TopP: 1, Seed: 1})
	const draws = 40000
	seen := make([]int, 4)
	for i := 0; i < draws; i++ {
		seen[s.Pick(logits())]++
	}
	for i, want := range []float64{0.5, 0.3, 0.15, 0.05} {
		got := float64(seen[i]) / draws
		if math.Abs(got-want) > 0.01 {
			t.Fatalf("token %d took %v of the draws, expected %v", i, got, want)
		}
	}
}

// A high temperature flattens the distribution, a low one sharpens it.
func TestTemperatureReshapesTheDistribution(t *testing.T) {
	const draws = 20000
	share := func(temp float32) float64 {
		s := New(Params{Temperature: temp, TopK: 0, TopP: 1, Seed: 2})
		count := 0
		for i := 0; i < draws; i++ {
			if s.Pick(logits()) == 0 {
				count++
			}
		}
		return float64(count) / draws
	}
	hot, cold := share(4), share(0.25)
	if hot > 0.4 {
		t.Fatalf("a temperature of 4 left the first token at %v", hot)
	}
	if cold < 0.85 { // p^4 renormalised puts it at 0.879
		t.Fatalf("a temperature of 0.25 left the first token at %v", cold)
	}
}

func TestASeedDeterminesTheRun(t *testing.T) {
	run := func(seed uint64) []int32 {
		s := New(Params{Temperature: 1, TopK: 0, TopP: 1, Seed: seed})
		out := make([]int32, 64)
		for i := range out {
			out[i] = s.Pick(logits())
		}
		return out
	}
	first, again, other := run(42), run(42), run(43)
	for i := range first {
		if first[i] != again[i] {
			t.Fatalf("the same seed diverged at draw %d", i)
		}
	}
	same := true
	for i := range first {
		if first[i] != other[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("two seeds produced the same run")
	}
}

// The defaults are the ones Gemma's own file carries.
func TestDefaults(t *testing.T) {
	d := Defaults()
	if d.Temperature != 1 || d.TopK != 64 || d.TopP != 0.95 {
		t.Fatalf("%+v", d)
	}
}

// A row the size of a real vocabulary, to say that the selection does not
// depend on the candidates being few.
func TestPickOverAWholeVocabulary(t *testing.T) {
	row := make([]float32, 262144)
	for i := range row {
		row[i] = float32(-i%1000) / 100
	}
	row[200000] = 50 // the one that should win outright
	s := New(Params{Temperature: 1, TopK: 64, TopP: 0.95, Seed: 9})
	for i := 0; i < 8; i++ {
		if got := s.Pick(row); got != 200000 {
			t.Fatalf("draw %d gave %d", i, got)
		}
	}
}
