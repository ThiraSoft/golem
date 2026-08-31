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
	for _, q := range []nn.Quant{nn.T3G, nn.T4G, nn.T5G} {
		name := q.String()
		g, ok := blockGeometry[name]
		if !ok {
			t.Fatalf("%s has no block geometry", name)
		}
		if g[0] != nn.T4GSeq {
			t.Fatalf("%s blocks %d weights, nn codes %d as one path", name, g[0], nn.T4GSeq)
		}
		if want := nn.T4GRowBytesN(nn.T4GSeq, q); g[1] != want {
			t.Fatalf("%s: the container reads %d bytes a block, nn writes %d", name, g[1], want)
		}
	}
}
