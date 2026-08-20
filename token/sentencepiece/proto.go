package sentencepiece

// Reading the `tokenizer.model` file, which is a protobuf message.
//
// The format is described by sentencepiece_model.proto; we need only two fields
// of it — the list of pieces and the way to normalize text — and protobuf can
// be read without a library: every field is preceded by a varint giving its
// number and its encoding. A hundred lines are enough, and the engine stays
// dependency-free.

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Piece kinds, as the proto numbers them.
const (
	typeNormal  = 1
	typeUnknown = 2
	typeControl = 3
	typeUser    = 4
	typeUnused  = 5
	typeByte    = 6
)

type piece struct {
	text  string
	score float32
	kind  int
}

// reader walks a protobuf message field by field.
type reader struct {
	data []byte
	pos  int
}

// field returns the number of the next field and its payload. Varints are
// decoded; strings and submessages are returned as they are.
func (r *reader) field() (number int, varint uint64, bytes []byte, err error) {
	if r.pos >= len(r.data) {
		return 0, 0, nil, nil // end of message
	}
	key, n := binary.Uvarint(r.data[r.pos:])
	if n <= 0 {
		return 0, 0, nil, fmt.Errorf("unreadable key at %d", r.pos)
	}
	r.pos += n
	number, encoding := int(key>>3), int(key&7)

	switch encoding {
	case 0: // varint
		v, n := binary.Uvarint(r.data[r.pos:])
		if n <= 0 {
			return 0, 0, nil, fmt.Errorf("unreadable integer at %d", r.pos)
		}
		r.pos += n
		return number, v, nil, nil
	case 2: // length-prefixed: string or submessage
		size, n := binary.Uvarint(r.data[r.pos:])
		if n <= 0 || r.pos+n+int(size) > len(r.data) {
			return 0, 0, nil, fmt.Errorf("unreadable block at %d", r.pos)
		}
		r.pos += n
		start := r.pos
		r.pos += int(size)
		return number, 0, r.data[start:r.pos], nil
	case 5: // 32 bits: the scores
		if r.pos+4 > len(r.data) {
			return 0, 0, nil, fmt.Errorf("truncated float at %d", r.pos)
		}
		v := uint64(binary.LittleEndian.Uint32(r.data[r.pos:]))
		r.pos += 4
		return number, v, nil, nil
	case 1: // 64 bits, unused here but skipped cleanly
		r.pos += 8
		return number, 0, nil, nil
	default:
		return 0, 0, nil, fmt.Errorf("unsupported encoding %d", encoding)
	}
}

// parse reads the model and returns its pieces and its normalization rules.
func parse(data []byte) ([]piece, normalizerSpec, error) {
	var pieces []piece
	rules := normalizerSpec{DummyPrefix: true, CollapseSpaces: true, EscapeSpaces: true}

	r := &reader{data: data}
	for {
		before := r.pos
		number, _, block, err := r.field()
		if err != nil {
			return nil, rules, err
		}
		if r.pos == before {
			break
		}
		switch number {
		case 1: // pieces
			p, err := parsePiece(block)
			if err != nil {
				return nil, rules, err
			}
			pieces = append(pieces, p)
		case 3: // normalizer_spec
			if err := rules.parse(block); err != nil {
				return nil, rules, err
			}
		}
	}
	return pieces, rules, nil
}

func parsePiece(block []byte) (piece, error) {
	p := piece{kind: typeNormal}
	r := &reader{data: block}
	for {
		before := r.pos
		number, v, bytes, err := r.field()
		if err != nil {
			return p, err
		}
		if r.pos == before {
			return p, nil
		}
		switch number {
		case 1:
			p.text = string(bytes)
		case 2:
			p.score = math.Float32frombits(uint32(v))
		case 3:
			p.kind = int(v)
		}
	}
}

// normalizerSpec holds the few normalizer flags that change the segmentation.
// The rest of the specification — the precompiled character map — is handled in
// normalize.go.
type normalizerSpec struct {
	DummyPrefix    bool // add_dummy_prefix: a leading space
	CollapseSpaces bool // remove_extra_whitespaces
	EscapeSpaces   bool // escape_whitespaces: the space becomes ▁
}

func (n *normalizerSpec) parse(block []byte) error {
	r := &reader{data: block}
	for {
		before := r.pos
		number, v, _, err := r.field()
		if err != nil {
			return err
		}
		if r.pos == before {
			return nil
		}
		switch number {
		case 3:
			n.DummyPrefix = v != 0
		case 4:
			n.CollapseSpaces = v != 0
		case 5:
			n.EscapeSpaces = v != 0
		}
	}
}
