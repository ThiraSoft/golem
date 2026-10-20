package yue2

import (
	"math"
	"slices"
	"sort"
	"testing"
)

func TestSamplerValidationAndRows(t *testing.T) {
	if err := ABCSampling.Valid(); err != nil {
		t.Errorf("ABCSampling invalid: %v", err)
	}
	if err := SemanticSampling.Valid(); err != nil {
		t.Errorf("SemanticSampling invalid: %v", err)
	}

	invalidCases := []Sampling{
		{Temperature: -0.1, TopP: 0.9, TopK: 10, RepetitionPenalty: 1.1, PenaltyWindow: 10, MinTokens: 1, MaxTokens: 10},
		{Temperature: 5.1, TopP: 0.9, TopK: 10, RepetitionPenalty: 1.1, PenaltyWindow: 10, MinTokens: 1, MaxTokens: 10},
		{Temperature: 0.7, TopP: 0.0, TopK: 10, RepetitionPenalty: 1.1, PenaltyWindow: 10, MinTokens: 1, MaxTokens: 10},
		{Temperature: 0.7, TopP: 1.1, TopK: 10, RepetitionPenalty: 1.1, PenaltyWindow: 10, MinTokens: 1, MaxTokens: 10},
		{Temperature: 0.7, TopP: 0.9, TopK: 0, RepetitionPenalty: 1.1, PenaltyWindow: 10, MinTokens: 1, MaxTokens: 10},
		{Temperature: 0.7, TopP: 0.9, TopK: 10, RepetitionPenalty: 0.0, PenaltyWindow: 10, MinTokens: 1, MaxTokens: 10},
		{Temperature: 0.7, TopP: 0.9, TopK: 10, RepetitionPenalty: 1.1, PenaltyWindow: 0, MinTokens: 1, MaxTokens: 10},
		{Temperature: 0.7, TopP: 0.9, TopK: 10, RepetitionPenalty: 1.1, PenaltyWindow: 101, MinTokens: 1, MaxTokens: 10},
		{Temperature: 0.7, TopP: 0.9, TopK: 10, RepetitionPenalty: 1.1, PenaltyWindow: 10, MinTokens: -1, MaxTokens: 10},
		{Temperature: 0.7, TopP: 0.9, TopK: 10, RepetitionPenalty: 1.1, PenaltyWindow: 10, MinTokens: 10, MaxTokens: 5},
	}
	for i, c := range invalidCases {
		if err := c.Valid(); err == nil {
			t.Errorf("case %d: expected validation error, got nil", i)
		}
	}

	row0ABC, rowsABC := PhaseRows(PhaseABC)
	if row0ABC != 0 || rowsABC != 151849 {
		t.Errorf("PhaseRows(PhaseABC) = (%d, %d), want (0, 151849)", row0ABC, rowsABC)
	}

	row0Sem, rowsSem := PhaseRows(PhaseSemantic)
	if row0Sem != 151852 || rowsSem != 32769 {
		t.Errorf("PhaseRows(PhaseSemantic) = (%d, %d), want (151852, 32769)", row0Sem, rowsSem)
	}
}

func TestDistributionAndDraw(t *testing.T) {
	f := loadFixtures(t)
	req := loadTestReplay(t)

	cases := []struct {
		name     string
		phase    Phase
		sampling Sampling
	}{
		{"score_0", PhaseABC, ABCSampling},
		{"score_1", PhaseABC, ABCSampling},
		{"score_2", PhaseABC, ABCSampling},
		{"semantic_0", PhaseSemantic, SemanticSampling},
		{"semantic_1", PhaseSemantic, SemanticSampling},
		{"semantic_2", PhaseSemantic, SemanticSampling},
		{"penalty_window", PhaseSemantic, SemanticSampling},
		{"with_end_token", PhaseABC, ABCSampling},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if !f.has("sample/" + tc.name + "_logits") {
				t.Skipf("sample/%s not recorded: see ref/yue2/README.md", tc.name)
			}

			logits := f.read(t, "sample/"+tc.name+"_logits")
			history := f.readInt32(t, "sample/"+tc.name+"_history")
			wantCandIDs := f.readInt32(t, "sample/"+tc.name+"_cand_ids")
			wantCandScores := f.read(t, "sample/"+tc.name+"_cand_scores")
			wantToken := f.readInt32(t, "sample/"+tc.name+"_token")[0]
			step := int(f.readInt32(t, "sample/"+tc.name+"_step")[0])

			var out []Candidate
			gotCandidates := Distribution(logits, tc.sampling, history, step, tc.phase, out)

			if len(gotCandidates) != len(wantCandIDs) {
				t.Fatalf("%s: candidates count mismatch: got %d, want %d",
					tc.name, len(gotCandidates), len(wantCandIDs))
			}

			for i := range gotCandidates {
				if gotCandidates[i].ID != wantCandIDs[i] {
					t.Fatalf("%s: candidate %d ID mismatch: got %d, want %d",
						tc.name, i, gotCandidates[i].ID, wantCandIDs[i])
				}
				diff := math.Abs(float64(gotCandidates[i].Score - wantCandScores[i]))
				denom := math.Max(math.Abs(float64(wantCandScores[i])), 1e-6)
				if diff/denom > 1e-6 {
					t.Errorf("%s: candidate %d score mismatch: got %g, want %g (rel %g)",
						tc.name, i, gotCandidates[i].Score, wantCandScores[i], diff/denom)
				}
			}

			gotToken := Draw(gotCandidates, req.LMSeed, int64(step))
			if gotToken != wantToken {
				t.Errorf("%s: drawn token mismatch: got %d, want %d", tc.name, gotToken, wantToken)
			}
		})
	}
}

// BenchmarkDistribution times one score phase step: the whole ABC window of a
// recorded head (151849 rows), a full penalty window of 100 tokens with
// repeats, and the candidate buffer carried from step to step as Generate does.
func BenchmarkDistribution(b *testing.B) {
	f := loadFixtures(b)
	logits := f.read(b, "sample/score_0_logits")
	history := make([]int32, 100)
	for i := range history {
		history[i] = int32((i * 37) % 61 * 97)
	}
	var out []Candidate
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out = Distribution(logits, ABCSampling, history, 100, PhaseABC, out)
	}
	b.ReportMetric(float64(b.Elapsed().Microseconds())/1e3/float64(b.N), "ms/step")
	if len(out) == 0 {
		b.Fatal("no candidate")
	}
}

// distributionBySort is the top-k and nucleus cut the way it reads in
// yue2.cpp's sampling.h with the whole window sorted first: every candidate at
// or above the k-th score survives. It takes the scores after penalty and
// temperature, which Distribution computes before cutting.
func distributionBySort(c []Candidate, k int, topP float32) []Candidate {
	c = slices.Clone(c)
	sort.SliceStable(c, func(i, j int) bool {
		if c[i].Score != c[j].Score {
			return c[i].Score > c[j].Score
		}
		return c[i].ID < c[j].ID
	})
	k = min(k, len(c))
	threshold := c[k-1].Score
	kept := k
	for kept < len(c) && c[kept].Score >= threshold {
		kept++
	}
	c = c[:kept]
	if topP >= 1 {
		return c
	}
	var sum, cumulative float64
	for i := range c {
		sum += math.Exp(float64(c[i].Score - c[0].Score))
	}
	for i := range c {
		if i > 0 && cumulative > float64(topP) {
			return c[:i]
		}
		cumulative += math.Exp(float64(c[i].Score-c[0].Score)) / sum
	}
	return c
}

// TestDistributionTies puts many candidates on the same few scores, so the
// k-th score is shared by several of them, and checks that the selection keeps
// exactly the set and order a sort of the whole window gives.
func TestDistributionTies(t *testing.T) {
	_, rows := PhaseRows(PhaseABC)
	logits := make([]float32, rows)
	rng := uint32(12345)
	next := func() uint32 { rng = rng*1664525 + 1013904223; return rng >> 8 }

	// The hand case first: three leaders, then four candidates tied at the
	// fifth score, then the rest below.
	for i := range logits {
		logits[i] = -10
	}
	for i, id := range []int{70, 30, 900} {
		logits[id] = float32(9 - i)
	}
	for _, id := range []int{5, 4000, 151000, 77} {
		logits[id] = 5
	}
	s := Sampling{Temperature: 1, TopP: 1, TopK: 5, RepetitionPenalty: 1, PenaltyWindow: 1, MinTokens: 0, MaxTokens: 1}
	got := Distribution(logits, s, nil, 0, PhaseABC, nil)
	want := []int32{70, 30, 900, 5, 77, 4000, 151000} // ties by id
	if !slices.Equal(candidateIDs(got), want) {
		t.Fatalf("hand case: kept %v, want %v", candidateIDs(got), want)
	}

	var out []Candidate
	for trial := 0; trial < 40; trial++ {
		for i := range logits {
			logits[i] = float32(int(next()%9)) - 8 // nine scores, thousands of ties on each
		}
		for i := 0; i < int(next()%20); i++ {
			logits[next()%uint32(rows)] = float32(next()%4) + 1 // a few leaders, tied too
		}
		s := ABCSampling
		s.TopK = 1 + int(next()%40)
		if trial%3 == 0 {
			s.TopP = 1
		}
		history := make([]int32, 100)
		for i := range history {
			history[i] = int32(next() % 300)
		}
		// The reference sees the same scores Distribution cuts: run it with no
		// cut at all (k as large as the window, no nucleus) and cut that.
		all := s
		all.TopK, all.TopP = rows, 1
		full := Distribution(logits, all, history, 100, PhaseABC, nil)
		want := distributionBySort(full, s.TopK, s.TopP)
		out = Distribution(logits, s, history, 100, PhaseABC, out)
		if !slices.Equal(out, want) {
			t.Fatalf("trial %d (top_k %d, top_p %g): kept %d candidates %v, want %d %v",
				trial, s.TopK, s.TopP, len(out), candidateIDs(out), len(want), candidateIDs(want))
		}
	}
}

func candidateIDs(c []Candidate) []int32 {
	ids := make([]int32, len(c))
	for i := range c {
		ids[i] = c[i].ID
	}
	return ids
}

// TestDistributionNaN is a head that answered NaN: the cut must still leave a
// candidate to draw, the best score that is a number.
func TestDistributionNaN(t *testing.T) {
	_, rows := PhaseRows(PhaseABC)
	logits := make([]float32, rows)
	nan := float32(math.NaN())
	for i := range logits {
		logits[i] = nan
	}
	logits[7], logits[900] = 1, 2
	got := Distribution(logits, ABCSampling, nil, 100, PhaseABC, nil)
	if len(got) == 0 || got[0].ID != 900 {
		t.Fatalf("NaN window with two numbers: kept %v, want 900 first", candidateIDs(got))
	}
	logits[7], logits[900] = nan, nan
	got = Distribution(logits, ABCSampling, nil, 100, PhaseABC, nil)
	if len(got) != 1 {
		t.Fatalf("all-NaN window: kept %d candidates, want 1", len(got))
	}
	_ = Draw(got, 1, 0)
}
