package nn

import "testing"

// The sections this checkpoint declares, and the geometry they belong to.
var qwen35Sections = Sections{11, 11, 10, 0}

// A position whose components are equal must give the scalar rotation exactly.
// Not nearly: the same float32s. Text depends on it, and so does every fixture
// recorded before M-RoPE existed.
func TestPrepareMultiDegeneratesToPrepare(t *testing.T) {
	var scalar, multi RoPETable
	for _, dims := range []int{64, 128} {
		for _, pos := range []int{0, 1, 7, 512, 4095} {
			scalar.Prepare(dims, pos, 1e7, nil)
			multi.PrepareMulti(dims, [4]int{pos, pos, pos, pos}, 1e7, qwen35Sections, nil)
			if len(scalar.Cos) != len(multi.Cos) {
				t.Fatalf("dims %d: %d entries against %d", dims, len(scalar.Cos), len(multi.Cos))
			}
			for i := range scalar.Cos {
				if scalar.Cos[i] != multi.Cos[i] || scalar.Sin[i] != multi.Sin[i] {
					t.Fatalf("dims %d, position %d, entry %d: (%v, %v) against (%v, %v)",
						dims, pos, i, scalar.Cos[i], scalar.Sin[i], multi.Cos[i], multi.Sin[i])
				}
			}
		}
	}
}

// Moving one component must move exactly the entries that component owns.
// This pins the bounds of the round robin; what the angle itself should be is
// ggml's to say, in TestMRoPEMatchesGGML.
//
// The base is 100 rather than the checkpoint's 1e7 on purpose: at 1e7 the last
// frequencies are so small that two nearby positions give the same float32
// cosine, and an entry that did move would look as though it had not.
func TestEachSectionMovesItsOwnEntries(t *testing.T) {
	const dims = 64
	sect := qwen35Sections[0] + qwen35Sections[1] + qwen35Sections[2] + qwen35Sections[3]

	var base RoPETable
	base.PrepareMulti(dims, [4]int{5, 5, 5, 5}, 100, qwen35Sections, nil)
	held := append([]float32(nil), base.Cos...)

	for _, c := range []struct {
		name  string
		index int
	}{{"t", 0}, {"h", 1}, {"w", 2}} {
		at := [4]int{5, 5, 5, 5}
		at[c.index] = 9
		var moved RoPETable
		moved.PrepareMulti(dims, at, 100, qwen35Sections, nil)
		for i := range held {
			sector := i % sect
			want := sector%3 == c.index && sector < 3*qwen35Sections[c.index]
			got := held[i] != moved.Cos[i]
			if got != want {
				t.Errorf("%s: entry %d (sector %d) moved=%v, wanted %v", c.name, i, sector, got, want)
			}
		}
	}
}

// Sections that sum to zero mean a checkpoint without M-RoPE, and must take
// the scalar path rather than divide by zero.
func TestNoSectionsIsTheScalarRotation(t *testing.T) {
	var scalar, multi RoPETable
	scalar.Prepare(64, 12, 1e7, nil)
	multi.PrepareMulti(64, [4]int{12, 99, 99, 99}, 1e7, Sections{}, nil)
	for i := range scalar.Cos {
		if scalar.Cos[i] != multi.Cos[i] || scalar.Sin[i] != multi.Sin[i] {
			t.Fatalf("entry %d: (%v, %v) against (%v, %v)",
				i, scalar.Cos[i], scalar.Sin[i], multi.Cos[i], multi.Sin[i])
		}
	}
}
