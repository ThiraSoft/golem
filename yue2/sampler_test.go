package yue2

import (
	"math"
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
