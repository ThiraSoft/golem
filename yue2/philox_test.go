package yue2

import (
	"testing"
)

func TestPhilox(t *testing.T) {
	f := loadFixtures(t)
	if !f.has("philox/counters") || !f.has("philox/seeds") || !f.has("philox/words") {
		t.Skip("philox fixtures not found: see ref/yue2/README.md")
	}

	countersRaw := f.readInt32(t, "philox/counters")
	seedsRaw := f.readInt32(t, "philox/seeds")
	wordsRaw := f.readInt32(t, "philox/words")

	numCases := len(countersRaw) / 4
	if numCases == 0 || len(seedsRaw) != numCases*2 || len(wordsRaw) != numCases*4 {
		t.Fatalf("mismatched philox fixture lengths: %d counters, %d seeds, %d words",
			len(countersRaw), len(seedsRaw), len(wordsRaw))
	}

	for i := 0; i < numCases; i++ {
		ctr := [4]uint32{
			uint32(countersRaw[i*4+0]),
			uint32(countersRaw[i*4+1]),
			uint32(countersRaw[i*4+2]),
			uint32(countersRaw[i*4+3]),
		}
		k0 := uint32(seedsRaw[i*2+0])
		k1 := uint32(seedsRaw[i*2+1])
		want := [4]uint32{
			uint32(wordsRaw[i*4+0]),
			uint32(wordsRaw[i*4+1]),
			uint32(wordsRaw[i*4+2]),
			uint32(wordsRaw[i*4+3]),
		}

		got := Philox4x32_10(ctr, k0, k1)
		if got != want {
			t.Errorf("case %d: got %v, want %v", i, got, want)
		}
	}
}
