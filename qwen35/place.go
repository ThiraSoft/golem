package qwen35

import "github.com/ThiraSoft/golem/vk"

// Place is one token's conversation, its entry in the cache, and where it sits
// on the model's three rotation axes.
//
// Pos and the triple are not the same number. Pos is where the key and the
// value are written and what a visible range is counted in; it rises by one per
// token, always. T, H and W drive the rotation and nothing else. Text has all
// four equal — which is why a scalar position served until an image needed to
// put several tokens at one T and let Pos go on rising.
//
// llama.cpp arranges this the other way round: its KV cell stores T as its
// position and which cell a token lands in is the allocator's choice. This
// engine indexes its cache by position directly, so the two numbers have to be
// told apart here rather than there.
type Place struct {
	Slot    int
	Pos     int
	T, H, W int
}

// Run is the places of a run of consecutive text positions in one slot, where
// every axis follows Pos. It is what reading a prompt and drawing an answer
// both offer.
func Run(slot, startPos, n int) []Place {
	places := make([]Place, n)
	for i := range places {
		p := startPos + i
		places[i] = Place{Slot: slot, Pos: p, T: p, H: p, W: p}
	}
	return places
}

// at is the rotation's four components, in the order nn.PrepareMulti reads
// them. The fourth is the axis a vision encoder would turn by, which no block
// of this model does; it travels because the file declares a width for it.
func (p Place) at() [4]int { return [4]int{p.T, p.H, p.W, 0} }

// Next is the place one token further on, which advances every axis. It is
// what text does, and what a token following an image does too: the picture is
// behind it, and what comes after a picture advances the way text always has.
// Only the tokens *inside* a span hold an axis still.
func (p Place) Next() Place {
	return Place{Slot: p.Slot, Pos: p.Pos + 1, T: p.T + 1, H: p.H + 1, W: p.W + 1}
}

// gpu is this place as the device path spells it. The card has no notion of a
// slot — the pipeline holds one conversation — so only the four numbers cross.
func (p Place) gpu() vk.QwenPlace {
	return vk.QwenPlace{Pos: p.Pos, T: p.T, H: p.H, W: p.W}
}
