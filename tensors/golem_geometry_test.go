package tensors

import (
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestGolemBlockGeometryMatchesNN is the join between the container and the
// codec. blockGeometry is what says how many bytes a row of a tensor occupies,
// and it is the only copy of that arithmetic outside nn — a reader that
// disagrees slices every row at the wrong offset and the model reads as noise,
// which is exactly what a layout change to one of these tiers produces if this
// table is not moved with it.
func TestGolemBlockGeometryMatchesNN(t *testing.T) {
	for _, q := range []nn.Quant{nn.H3G, nn.H4G} {
		p, name := nn.PairTierOf(q), q.String()
		if g := blockGeometry[name]; g[0] != nn.GolemSeq || g[1] != p.RowBytes(nn.GolemSeq) {
			t.Fatalf("%s: the container reads %v, nn writes %d bytes a %d-weight block", name, g, p.RowBytes(nn.GolemSeq), nn.GolemSeq)
		}
		if golemPairState[name] != p.L {
			t.Fatalf("%s: the container says the state is %d bits, nn says %d", name, golemPairState[name], p.L)
		}
	}
}
