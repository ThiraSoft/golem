package ugm

// The two lookup structures the tokenizer walks a byte at a time.

import (
	"encoding/binary"
	"fmt"
	"sort"
)

// trie is llama.cpp's naive_trie, kept compact. The vocabulary has a quarter
// of a million pieces and a map per node would hold them in some sixty
// megabytes; here every node's children sit side by side, sorted by byte, in
// three flat arrays.
type trie struct {
	first  []int32 // node n's children are labels[first[n]:first[n+1]]
	labels []byte
	child_ []int32
	value  []int // the piece that ends at a node, or -1

	building map[uint64]int32 // parent<<8|byte -> child, until the first lookup
}

func (t *trie) insert(s string, id int) {
	if t.value == nil {
		t.value = []int{-1}
		t.building = map[uint64]int32{}
	}
	if t.building == nil {
		panic("ugm: a trie cannot grow once it has been read")
	}
	node := int32(0)
	for i := 0; i < len(s); i++ {
		key := uint64(node)<<8 | uint64(s[i])
		next, ok := t.building[key]
		if !ok {
			next = int32(len(t.value))
			t.value = append(t.value, -1)
			t.building[key] = next
		}
		node = next
	}
	// A piece written twice keeps the later identifier, as naive_trie does.
	t.value[node] = id
}

// seal lays the children out flat and drops the map.
func (t *trie) seal() {
	if t.building == nil {
		return
	}
	type edge struct {
		parent int32
		label  byte
		child  int32
	}
	edges := make([]edge, 0, len(t.building))
	for k, c := range t.building {
		edges = append(edges, edge{int32(k >> 8), byte(k), c})
	}
	sort.Slice(edges, func(a, b int) bool {
		if edges[a].parent != edges[b].parent {
			return edges[a].parent < edges[b].parent
		}
		return edges[a].label < edges[b].label
	})
	t.first = make([]int32, len(t.value)+1)
	t.labels = make([]byte, len(edges))
	t.child_ = make([]int32, len(edges))
	for i, e := range edges {
		t.first[e.parent+1]++
		t.labels[i] = e.label
		t.child_[i] = e.child
	}
	for n := 1; n < len(t.first); n++ {
		t.first[n] += t.first[n-1]
	}
	t.building = nil
}

// child is the node reached from node by b, or -1.
func (t *trie) child(node int, b byte) int {
	if t.value == nil {
		return -1
	}
	t.seal()
	lo, hi := int(t.first[node]), int(t.first[node+1])
	for lo < hi {
		mid := (lo + hi) / 2
		switch l := t.labels[mid]; {
		case l == b:
			return int(t.child_[mid])
		case l < b:
			lo = mid + 1
		default:
			hi = mid
		}
	}
	return -1
}

// longest is the length of the longest piece s begins with, or 0.
func (t *trie) longest(s string) int {
	n, node := 0, 0
	for i := 0; i < len(s); i++ {
		if node = t.child(node, s[i]); node < 0 {
			break
		}
		if t.value[node] >= 0 {
			n = i + 1
		}
	}
	return n
}

// charsmap is SentencePiece's precompiled normalization map: a trie over the
// bytes of the input, stored as an XOR-compressed compact double array
// (Kanda 2018), whose leaves point into a block of NUL-terminated
// replacements. The layout is the file's own:
//
//	u32 length of the array in bytes | the array | the replacements
type charsmap struct {
	xcda []uint32
	repl []byte
}

func parseCharsmap(blob []byte) (charsmap, error) {
	if len(blob) < 4 {
		return charsmap{}, fmt.Errorf("ugm: a character map of %d bytes", len(blob))
	}
	size := int(binary.LittleEndian.Uint32(blob))
	if size%4 != 0 || 4+size > len(blob) {
		return charsmap{}, fmt.Errorf("ugm: the character map announces %d bytes of %d", size, len(blob)-4)
	}
	m := charsmap{xcda: make([]uint32, size/4), repl: blob[4+size:]}
	for i := range m.xcda {
		m.xcda[i] = binary.LittleEndian.Uint32(blob[4+4*i:])
	}
	return m, nil
}

// Each entry packs a node: BASE in bits 10-30, shifted left by 8 more when bit
// 9 is set; LEAF in bit 8; LCHECK in bits 0-7, with bit 31 set on the entries
// that hold a replacement's offset instead, so that they never pass for a
// child.
func (m charsmap) base(i uint32) uint32   { n := m.xcda[i]; return (n >> 10) << ((n & (1 << 9)) >> 6) }
func (m charsmap) lcheck(i uint32) uint32 { return m.xcda[i] & (1<<31 | 0xff) }
func (m charsmap) leaf(i uint32) bool     { return m.xcda[i]>>8&1 != 0 }
func (m charsmap) value(i uint32) uint32  { return m.xcda[i] & (1<<31 - 1) }

// longest is the replacement for the longest prefix of s the map knows, and
// the length of that prefix; 0 when it knows none.
func (m charsmap) longest(s string) (string, int) {
	if len(m.xcda) == 0 {
		return "", 0
	}
	length, offset := 0, uint32(0)
	node := m.base(0)
	for i := 0; i < len(s); i++ {
		c := uint32(s[i])
		if c == 0 {
			break
		}
		node ^= c
		if int(node) >= len(m.xcda) || m.lcheck(node) != c {
			break
		}
		leaf := m.leaf(node)
		node ^= m.base(node)
		if int(node) >= len(m.xcda) {
			break
		}
		if leaf {
			length = i + 1
			offset = m.value(node)
		}
	}
	if length == 0 || int(offset) >= len(m.repl) {
		return "", 0
	}
	end := int(offset)
	for end < len(m.repl) && m.repl[end] != 0 {
		end++
	}
	return string(m.repl[offset:end]), length
}
