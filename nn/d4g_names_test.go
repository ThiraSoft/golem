package nn

import "testing"

// The converter writes these names and every reader looks for them, so the two
// sides share this function rather than each keeping a copy of the convention.
// A drift between them is a model that loads and answers nonsense.
func TestD4GVectorNames(t *testing.T) {
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
		got := D4GVectorNames(c.tensor)
		if len(got) == 0 || got[0] != c.want {
			t.Errorf("%s looks for %v, want %s first", c.tensor, got, c.want)
		}
	}
	// Every name has a fallback under the tensor's own, for a converter that
	// had no site to file it under.
	if n := D4GVectorNames("blk.1.attn_v.weight"); len(n) != 2 || n[1] != "blk.1.attn_v.weight.pre" {
		t.Errorf("no fallback: %v", n)
	}
}
