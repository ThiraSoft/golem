package sentencepiece

// SentencePiece tokenizer, unigram model.
//
// The principle: the vocabulary gives each piece a log-probability, and the
// chosen segmentation is the one maximizing the sum of the scores. It is a
// shortest path through the lattice of possible segmentations, which dynamic
// programming solves in a single sweep.

import (
	"math"
	"os"
	"strings"
	"unicode/utf8"
)

// Space is the character SentencePiece uses to represent the space, so that the
// pieces keep track of word boundaries.
const Space = "▁"

// Tokenizer splits text into identifiers.
type Tokenizer struct {
	pieces    []piece
	index     map[string]int
	maxLength int
	rules     normalizerSpec
	minScore  float32
	byteIDs   [256]int // identifiers of the <0xNN> pieces, for the unknowns
	unknown   int
}

// Load reads a tokenizer.model file.
func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	pieces, rules, err := parse(data)
	if err != nil {
		return nil, err
	}

	t := &Tokenizer{
		pieces:   pieces,
		index:    make(map[string]int, len(pieces)),
		rules:    rules,
		minScore: float32(math.Inf(1)),
	}
	for i := range t.byteIDs {
		t.byteIDs[i] = -1
	}
	for id, p := range pieces {
		switch p.kind {
		case typeNormal, typeUser:
			// First one wins: the vocabulary has no duplicates, but better not
			// to depend on that.
			if _, seen := t.index[p.text]; !seen {
				t.index[p.text] = id
			}
			if n := len(p.text); n > t.maxLength {
				t.maxLength = n
			}
			if p.score < t.minScore {
				t.minScore = p.score
			}
		case typeByte:
			if v, ok := byteValue(p.text); ok {
				t.byteIDs[v] = id
			}
		case typeUnknown:
			t.unknown = id
		}
	}
	return t, nil
}

// Size is the number of identifiers in the vocabulary.
func (t *Tokenizer) Size() int { return len(t.pieces) }

// Piece returns the written form of an identifier.
func (t *Tokenizer) Piece(id int) string { return t.pieces[id].text }

// byteValue recognizes the "<0x41>" fallback pieces.
func byteValue(s string) (byte, bool) {
	if len(s) != 6 || !strings.HasPrefix(s, "<0x") || s[5] != '>' {
		return 0, false
	}
	var v byte
	for _, c := range []byte(s[3:5]) {
		switch {
		case c >= '0' && c <= '9':
			v = v<<4 | (c - '0')
		case c >= 'A' && c <= 'F':
			v = v<<4 | (c - 'A' + 10)
		case c >= 'a' && c <= 'f':
			v = v<<4 | (c - 'a' + 10)
		default:
			return 0, false
		}
	}
	return v, true
}

// Encode returns the identifiers of the text.
func (t *Tokenizer) Encode(text string) []int {
	normalized := t.normalize(text)
	if normalized == "" {
		return nil
	}

	n := len(normalized)
	// best[i]: score of the best segmentation of the first i bytes.
	best := make([]float64, n+1)
	origin := make([]int, n+1) // start of the last piece
	pieceID := make([]int, n+1)
	for i := 1; i <= n; i++ {
		best[i] = math.Inf(-1)
		pieceID[i] = -1
	}

	// A character never seen before must remain traversable, otherwise a single
	// exotic symbol would make the whole sentence unencodable. It then costs
	// markedly more than the worst known piece.
	unknownPenalty := float64(t.minScore) - 10

	for start := 0; start < n; start++ {
		if math.IsInf(best[start], -1) {
			continue
		}
		found := false
		end := start + t.maxLength
		if end > n {
			end = n
		}
		for j := start + 1; j <= end; j++ {
			id, ok := t.index[normalized[start:j]]
			if !ok {
				continue
			}
			found = true
			score := best[start] + float64(t.pieces[id].score)
			if score > best[j] {
				best[j], origin[j], pieceID[j] = score, start, id
			}
		}
		if !found {
			// fall back on a whole character, re-emitted as bytes on readback
			_, size := utf8.DecodeRuneInString(normalized[start:])
			j := start + size
			score := best[start] + unknownPenalty
			if score > best[j] {
				best[j], origin[j], pieceID[j] = score, start, -1
			}
		}
	}

	// Walk the path back, then put it the right way round.
	var segments [][2]int
	var ids []int
	for i := n; i > 0; {
		segments = append(segments, [2]int{origin[i], i})
		ids = append(ids, pieceID[i])
		i = origin[i]
	}

	var out []int
	for k := len(ids) - 1; k >= 0; k-- {
		if ids[k] >= 0 {
			out = append(out, ids[k])
			continue
		}
		// Unknown piece: each of its bytes takes its fallback piece.
		for _, b := range []byte(normalized[segments[k][0]:segments[k][1]]) {
			if id := t.byteIDs[b]; id >= 0 {
				out = append(out, id)
			} else {
				out = append(out, t.unknown)
			}
		}
	}
	return out
}

// Decode reconstructs the text, for error messages and tests.
func (t *Tokenizer) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.pieces) {
			continue
		}
		p := t.pieces[id]
		if p.kind == typeByte {
			if v, ok := byteValue(p.text); ok {
				b.WriteByte(v)
			}
			continue
		}
		b.WriteString(p.text)
	}
	return strings.TrimPrefix(strings.ReplaceAll(b.String(), Space, " "), " ")
}
