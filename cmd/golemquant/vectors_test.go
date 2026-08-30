package main

import (
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

// A hybrid's block, as this converter writes it: four input projections and an
// output one, none of which a full attention has, beside the two vectors the
// calibration measured for the sites they read.
func hybridBlock(rotated func(string) string) ([]tensors.OutStream, map[string]string) {
	mats := map[string]int{
		"blk.3.attn_qkv.weight":  5120,
		"blk.3.attn_gate.weight": 5120,
		"blk.3.ssm_alpha.weight": 5120,
		"blk.3.ssm_beta.weight":  5120,
		"blk.3.ssm_out.weight":   6144,
	}
	out := []tensors.OutStream{
		{Name: "blk.3.qkv.pre", Shape: []int{5120}, DType: "F32"},
		{Name: "blk.3.o.pre", Shape: []int{6144}, DType: "F32"},
	}
	by := map[string]string{}
	for name, cols := range mats {
		out = append(out, tensors.OutStream{Name: name, Shape: []int{cols}, DType: "D4G"})
		by[name] = rotated(name)
		if strings.HasSuffix(by[name], ".weight.pre") {
			out = append(out, tensors.OutStream{Name: by[name], Shape: []int{cols}, DType: "F32"})
		}
	}
	return out, by
}

// What a converter that does not know these five names does: no site, so each
// matrix is rotated blind and its vector written under its own name — while the
// loader, which does know them, reads the site's instead. Every shape agrees
// and the model answers nonsense, so only this check stands between the two.
func TestBlindRotationShadowedByItsSiteIsRefused(t *testing.T) {
	out, by := hybridBlock(func(name string) string { return name + ".pre" })
	err := checkVectors(out, by)
	if err == nil {
		t.Fatal("a blind rotation the loader would read past was accepted")
	}
	if !strings.Contains(err.Error(), "blk.3.qkv.pre") && !strings.Contains(err.Error(), "blk.3.o.pre") {
		t.Errorf("the error does not name the vector the loader would read: %v", err)
	}
}

// And what it does once it asks nn which site each name reads.
func TestSiteRotationIsWhatTheLoaderReads(t *testing.T) {
	out, by := hybridBlock(func(name string) string {
		if strings.Contains(name, "ssm_out") {
			return "blk.3.o.pre"
		}
		return "blk.3.qkv.pre"
	})
	if err := checkVectors(out, by); err != nil {
		t.Fatal(err)
	}
}

// A matrix no site names keeps its own vector, and nothing shadows it.
func TestOwnVectorIsReadWhenNoSiteExists(t *testing.T) {
	out := []tensors.OutStream{
		{Name: "blk.64.nextn.eh_proj.weight", Shape: []int{10240}, DType: "D4G"},
		{Name: "blk.64.nextn.eh_proj.weight.pre", Shape: []int{10240}, DType: "F32"},
		{Name: "blk.64.attn_q.weight", Shape: []int{5120}, DType: "D4G"},
		{Name: "blk.64.attn_q.weight.pre", Shape: []int{5120}, DType: "F32"},
	}
	by := map[string]string{
		"blk.64.nextn.eh_proj.weight": "blk.64.nextn.eh_proj.weight.pre",
		"blk.64.attn_q.weight":        "blk.64.attn_q.weight.pre",
	}
	if err := checkVectors(out, by); err != nil {
		t.Fatal(err)
	}
}

// A matrix left unrotated must find no vector at all.
func TestUnrotatedMatrixMustFindNothing(t *testing.T) {
	out := []tensors.OutStream{
		{Name: "blk.3.ffn_down.weight", Shape: []int{17408}, DType: "D4G"},
		{Name: "blk.3.down.pre", Shape: []int{17408}, DType: "F32"},
	}
	if err := checkVectors(out, map[string]string{"blk.3.ffn_down.weight": ""}); err == nil {
		t.Fatal("an unrotated matrix was allowed to pick up a site's vector")
	}
}

// A site nothing was measured at is still a site. The reader asks
// nn.D4GVectorNames, which offers a site's name before a tensor's own, so
// every matrix of that site has to be rotated by one vector filed under it —
// not by three of its own that the reader will never look at.
//
// This is the prediction block: it lies past the trunk a corpus walks, so no
// calibration reaches it, and its attention has the same three projections a
// measured block's does.
func TestAnUnmeasuredSiteStillHasOneVector(t *testing.T) {
	cols := 5120
	out := []tensors.OutStream{
		{Name: "blk.64.qkv.pre", Shape: []int{cols}, DType: "F32"},
	}
	by := map[string]string{}
	for _, n := range []string{"blk.64.attn_q.weight", "blk.64.attn_k.weight", "blk.64.attn_v.weight"} {
		out = append(out, tensors.OutStream{Name: n, Shape: []int{cols}, DType: "D4G"})
		by[n] = "blk.64.qkv.pre"
	}
	if err := checkVectors(out, by); err != nil {
		t.Fatal(err)
	}

	// And what it looked like before: a vector each, under names the reader
	// never reaches because the site's is offered first.
	out = []tensors.OutStream{{Name: "blk.64.qkv.pre", Shape: []int{cols}, DType: "F32"}}
	by = map[string]string{}
	for _, n := range []string{"blk.64.attn_q.weight", "blk.64.attn_k.weight", "blk.64.attn_v.weight"} {
		out = append(out,
			tensors.OutStream{Name: n, Shape: []int{cols}, DType: "D4G"},
			tensors.OutStream{Name: n + ".pre", Shape: []int{cols}, DType: "F32"})
		by[n] = n + ".pre"
	}
	if checkVectors(out, by) == nil {
		t.Fatal("three vectors under three names, and the reader would read one: accepted")
	}
}

// A probe names tensors, and a name is fields rather than letters.
func TestKeptNamesTensorsAndNotSubstrings(t *testing.T) {
	const list = "output.weight,token_embd.weight,ssm_alpha"
	for _, c := range []struct {
		name string
		want bool
	}{
		{"output.weight", true},
		{"token_embd.weight", true},
		{"blk.3.ssm_alpha.weight", true},
		// The one that made a probe answer about something else entirely.
		{"blk.3.attn_output.weight", false},
		{"blk.3.ffn_down.weight", false},
		{"output_norm.weight", false},
	} {
		if got := kept(c.name, list); got != c.want {
			t.Errorf("kept(%q) = %v, want %v", c.name, got, c.want)
		}
	}
}
