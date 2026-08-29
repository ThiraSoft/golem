package qwen35

import (
	"fmt"

	"github.com/ThiraSoft/golem/vk"
)

// The markers this checkpoint writes a picture with. They are read by name
// rather than by number: the numbers are this vocabulary's and the names are
// the family's.
const (
	VisionStart = "<|vision_start|>"
	VisionEnd   = "<|vision_end|>"
	ImagePad    = "<|image_pad|>"
)

// Prompt is a tokenized conversation with the pictures already in it.
//
// Tokens is what the cache holds — the markers included, and one ImagePad per
// row of every picture. Embeds is nil at every position but those, where it is
// the tower's own row. Images says where each picture sits and what grid it
// came from, which is what the positions are built from.
type Prompt struct {
	Tokens []int32
	Embeds [][]float32
	Images []ImageSpan
}

// ImageSpan is one picture inside a prompt: where its rows begin, how many
// there are, and the merged grid they were laid out on.
type ImageSpan struct {
	Start int
	Count int
	Cols  int // the merged grid's width, which is what a row index divides by
	Rows  int
}

// BuildPrompt splices encoded pictures into an encoded prompt.
//
// tokens is the rendered conversation as the vocabulary encoded it: the
// template wrote an empty VisionStart/VisionEnd pair for each picture, and
// this fills each pair in, in order, with one ImagePad and one row per output
// token of the tower.
//
// The template does not have to know how many rows a picture became, which is
// the point of writing the pair empty — the count is the tower's answer and is
// not known when the conversation is rendered. It is gemma/visionprompt.go's
// arrangement, for the same reason.
func (m *Model) BuildPrompt(tokens []int32, images [][][]float32, grids [][2]int) (*Prompt, error) {
	if len(images) != len(grids) {
		return nil, fmt.Errorf("qwen35: %d pictures and %d grids", len(images), len(grids))
	}
	start, pad, end, ok := m.visionMarkers()
	if !ok {
		return nil, fmt.Errorf("qwen35: this vocabulary has no vision markers")
	}

	p := &Prompt{}
	next := 0
	for i := 0; i < len(tokens); i++ {
		p.Tokens = append(p.Tokens, tokens[i])
		p.Embeds = append(p.Embeds, nil)
		if tokens[i] != start {
			continue
		}
		if i+1 >= len(tokens) || tokens[i+1] != end {
			return nil, fmt.Errorf("qwen35: a %s at %d is not closed by a %s", VisionStart, i, VisionEnd)
		}
		if next >= len(images) {
			return nil, fmt.Errorf("qwen35: the prompt opens more pictures than were encoded")
		}
		rows, grid := images[next], grids[next]
		if grid[0]*grid[1] != len(rows) {
			return nil, fmt.Errorf("qwen35: picture %d is a %dx%d grid, which is not %d rows",
				next, grid[0], grid[1], len(rows))
		}
		p.Images = append(p.Images, ImageSpan{
			Start: len(p.Tokens), Count: len(rows), Cols: grid[0], Rows: grid[1],
		})
		for _, row := range rows {
			p.Tokens = append(p.Tokens, pad)
			p.Embeds = append(p.Embeds, row)
		}
		next++
	}
	if next != len(images) {
		return nil, fmt.Errorf("qwen35: %d pictures encoded and %d placed", len(images), next)
	}
	m.grids = m.grids[:0]
	return p, nil
}

// Places says where each token of the prompt goes, starting at startPos.
//
// Text advances on every axis at once. A picture does not: every one of its
// rows shares the time the picture began at, and spreads over the height and
// the width of the grid it came from. The cache index rises by one throughout,
// which is what lets a picture occupy as many entries as it has rows while
// holding one time.
//
// This is mtmd_image_tokens_get_decoder_pos under MTMD_POS_TYPE_MROPE, where
// t is the position the picture starts at, y is that plus the row and x that
// plus the column.
func (p *Prompt) Places(slot, startPos int) []Place {
	at := make([]Place, len(p.Tokens))
	for i := range at {
		q := startPos + i
		at[i] = Place{Slot: slot, Pos: q, T: q, H: q, W: q}
	}
	for _, im := range p.Images {
		base := startPos + im.Start
		for r := 0; r < im.Count; r++ {
			at[im.Start+r] = Place{
				Slot: slot,
				Pos:  base + r,
				T:    base,
				H:    base + r/im.Cols,
				W:    base + r%im.Cols,
			}
		}
	}
	return at
}

// ForwardPrompt reads the whole prompt in one sweep and returns one hidden
// state per position.
//
// On the card it goes through the same pipeline a text run does: a picture's
// rows are seeded where the token table would have been read, and the axes
// travel in the position buffer. Nothing about the device path had to change
// for this — a picture keeps the cache index consecutive, which is the only
// thing that pass ever asked of a position.
func (m *Model) ForwardPrompt(p *Prompt, startPos int) [][]float32 {
	at := p.Places(m.slot, startPos)
	out := make([][]float32, len(p.Tokens))

	if m.gpuPipe != nil {
		embeds := make([][]float32, m.gpuPipe.WidthFor(len(p.Tokens)))
		for i := range embeds {
			embeds[i] = make([]float32, m.Cfg.Dim)
		}
		for t := 0; t < len(p.Tokens); {
			n := m.gpuPipe.WidthFor(len(p.Tokens) - t)
			places := make([]vk.QwenPlace, n)
			for c := 0; c < n; c++ {
				if row := p.Embeds[t+c]; row != nil {
					copy(embeds[c], row)
				} else {
					m.W.TokenEmbd.Row(int(p.Tokens[t+c]), embeds[c])
				}
				places[c] = at[t+c].gpu()
			}
			hs, err := m.gpuPipe.ForwardPlaces(embeds[:n], places)
			if err != nil {
				panic(fmt.Sprintf("qwen35: the GPU pipeline failed at position %d: %v", startPos+t, err))
			}
			for c := 0; c < n; c++ {
				out[t+c] = append([]float32(nil), hs[c]...)
			}
			copy(m.x, m.gpuPipe.HiddenColumn(n-1))
			t += n
		}
		return out
	}

	for t := range p.Tokens {
		if row := p.Embeds[t]; row != nil {
			out[t] = m.stepEmbedded(row, at[t])
		} else {
			out[t] = m.step(p.Tokens[t], at[t])
		}
	}
	return out
}

// Slice is the run [from, to) as a prompt of its own, with the pictures that
// fall inside it moved to match.
//
// A picture may be cut across two passes here, unlike Gemma's: this model
// attends causally over a picture as over text, so a row reads only what came
// before it and a cut costs nothing. What must not move is a row's axes, and
// those travel with the span rather than with the offset.
func (p *Prompt) Slice(from, to int) *Prompt {
	out := &Prompt{
		Tokens: p.Tokens[from:to],
		Embeds: p.Embeds[from:to],
	}
	for _, im := range p.Images {
		end := im.Start + im.Count
		if end <= from || im.Start >= to {
			continue
		}
		// The span keeps its original start so that a row's row-and-column
		// stay what the whole picture gave them; only the offset moves.
		out.Images = append(out.Images, ImageSpan{
			Start: im.Start - from,
			Count: im.Count,
			Cols:  im.Cols,
			Rows:  im.Rows,
		})
	}
	return out
}

// visionMarkers are the three identifiers a picture is written with, or false
// when the vocabulary carries none of them.
func (m *Model) visionMarkers() (start, pad, end int32, ok bool) {
	if m.visionStartID == 0 || m.imagePadID == 0 || m.visionEndID == 0 {
		return 0, 0, 0, false
	}
	return m.visionStartID, m.imagePadID, m.visionEndID, true
}

// SetVisionMarkers tells the model which identifiers stand for a picture. The
// vocabulary is opened beside the model rather than inside it, so whoever
// opened it is the one that knows.
func (m *Model) SetVisionMarkers(start, pad, end int32) {
	m.visionStartID, m.imagePadID, m.visionEndID = start, pad, end
}

// HasVisionMarkers reports whether the three are known.
func (m *Model) HasVisionMarkers() bool {
	_, _, _, ok := m.visionMarkers()
	return ok
}

// ForwardEmbeddedPlaces is one pass over tokens whose embeddings may be given
// rather than looked up, at places that need not follow the cache index.
//
// It is what a server's batched pass reaches. On a card this model holds one
// conversation — the delta net's state is a matrix a head and not a ring, so
// there is nothing to cut into slots — and the pass is therefore one run of
// consecutive cache positions, which is all the pipeline ever asked.
func (m *Model) ForwardEmbeddedPlaces(tokens []int32, embeds [][]float32, at []Place) [][]float32 {
	out := make([][]float32, len(tokens))

	if m.gpuPipe != nil {
		xs := make([][]float32, m.gpuPipe.WidthFor(len(tokens)))
		for i := range xs {
			xs[i] = make([]float32, m.Cfg.Dim)
		}
		for t := 0; t < len(tokens); {
			n := m.gpuPipe.WidthFor(len(tokens) - t)
			places := make([]vk.QwenPlace, n)
			for c := 0; c < n; c++ {
				if embeds != nil && embeds[t+c] != nil {
					copy(xs[c], embeds[t+c])
				} else {
					m.W.TokenEmbd.Row(int(tokens[t+c]), xs[c])
				}
				places[c] = at[t+c].gpu()
			}
			hs, err := m.gpuPipe.ForwardPlaces(xs[:n], places)
			if err != nil {
				panic(fmt.Sprintf("qwen35: the GPU pipeline failed at position %d: %v", at[t].Pos, err))
			}
			for c := 0; c < n; c++ {
				out[t+c] = append([]float32(nil), hs[c]...)
			}
			copy(m.x, m.gpuPipe.HiddenColumn(n-1))
			t += n
		}
		return out
	}

	for t := range tokens {
		m.UseSlot(at[t].Slot)
		if embeds != nil && embeds[t] != nil {
			out[t] = m.stepEmbedded(embeds[t], at[t])
		} else {
			out[t] = m.step(tokens[t], at[t])
		}
	}
	return out
}
