package nn

import "testing"

// The converter writes these names and every reader looks for them, so the two
// sides share this function rather than each keeping a copy of the convention.
// A drift between them is a model that loads and answers nonsense.
func TestGolemVectorNames(t *testing.T) {
	for _, c := range []struct{ tensor, want string }{
		{"blk.7.attn_k.weight", "blk.7.qkv.pre"},
		{"blk.7.attn_q.weight", "blk.7.qkv.pre"},
		{"blk.7.attn_output.weight", "blk.7.o.pre"},
		{"blk.0.ffn_up.weight", "blk.0.gateup.pre"},
		{"blk.31.ffn_down.weight", "blk.31.down.pre"},
		{"blk.3.ffn_gate_up_exps.weight", "blk.3.gateup_exps.pre"},
		{"blk.3.ffn_down_exps.weight", "blk.3.down_exps.pre"},
		{"token_embd.weight", "output.pre"},
		{"something.else.weight", "something.else.weight.pre"},
	} {
		got := GolemVectorNames(c.tensor)
		if len(got) == 0 || got[0] != c.want {
			t.Errorf("%s looks for %v, want %s first", c.tensor, got, c.want)
		}
	}
	// Every name has a fallback under the tensor's own, for a converter that
	// had no site to file it under.
	if n := GolemVectorNames("blk.1.attn_v.weight"); len(n) != 2 || n[1] != "blk.1.attn_v.weight.pre" {
		t.Errorf("no fallback: %v", n)
	}
}

// The converter asks which site a matrix reads and the loader asks what that
// site's vector is called; both questions go to the same table, so a matrix
// the converter treats as siteless can never be one the loader files under a
// site. A hybrid's linear-attention block is where the two drifted apart.
func TestGolemSiteAgreesWithTheNames(t *testing.T) {
	for _, c := range []struct{ matrix, site string }{
		{"attn_qkv", "qkv"}, {"attn_gate", "qkv"},
		{"ssm_alpha", "qkv"}, {"ssm_beta", "qkv"},
		{"ssm_out", "o"},
		{"attn_q", "qkv"}, {"attn_output", "o"},
		{"ffn_up", "gateup"}, {"ffn_down", "down"},
	} {
		site, ok := GolemSite(c.matrix)
		if !ok || site != c.site {
			t.Errorf("GolemSite(%q) = %q, %v; want %q", c.matrix, site, ok, c.site)
		}
		want := "blk.5." + c.site + ".pre"
		if got := GolemVectorNames("blk.5." + c.matrix + ".weight"); got[0] != want {
			t.Errorf("%s is filed under %s but read from %s", c.matrix, want, got[0])
		}
	}
	if _, ok := GolemSite("ssm_conv1d"); ok {
		t.Error("ssm_conv1d is not a matrix with a site")
	}
}

// The logit head is one site whether the model ties it to the table or keeps a
// matrix of its own. Both read what the final norm made, so both are filed
// under output.pre — and an untied model has the two of them there.
func TestTheHeadIsOneSiteTiedOrNot(t *testing.T) {
	for _, tensor := range []string{"token_embd.weight", "output.weight"} {
		got := GolemVectorNames(tensor)
		if len(got) != 2 || got[0] != "output.pre" || got[1] != tensor+".pre" {
			t.Errorf("%s reads %v, want [output.pre %s.pre]", tensor, got, tensor)
		}
	}
	// And output_norm.weight is not the head, however much its name looks it.
	if got := GolemVectorNames("output_norm.weight"); got[0] != "output_norm.weight.pre" {
		t.Errorf("output_norm.weight reads %v", got)
	}
}
