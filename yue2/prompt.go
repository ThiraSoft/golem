package yue2

import (
	"fmt"
)

// Cot is the chain of thought mode controlling how the ABC score plan reaches the AR backbone.
type Cot int

const (
	CotOff Cot = iota
	CotMelody
	CotFull
)

// ParseCot converts a mode name string ("full", "melody", "off") to a Cot value.
func ParseCot(s string) (Cot, error) {
	switch s {
	case "full":
		return CotFull, nil
	case "melody":
		return CotMelody, nil
	case "off":
		return CotOff, nil
	default:
		return CotOff, fmt.Errorf("unknown cot mode %q", s)
	}
}

// instruction returns the conditioning prompt instruction text for the given Cot mode.
func instruction(cot Cot) string {
	switch cot {
	case CotOff:
		return "Generate music with codec tokens from the given conditions."
	case CotMelody:
		return "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."
	default:
		return "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
	}
}

// requestText formats the conditioning text block with instruction, tags, and lyrics.
func requestText(cot Cot, style, lyrics string) string {
	return instruction(cot) + "\n[Tags]\n" + style + "\n[Lyrics]\n" + lyrics + "\n"
}

// DefaultGuidance returns the default classifier-free guidance scale for the Cot mode.
// In CotOff mode, guidance defaults to 1.01; melody and full modes default to 1.0.
func DefaultGuidance(cot Cot) float32 {
	if cot == CotOff {
		return 1.01
	}
	return 1.0
}

// BuildPrompt assembles the conditional token sequence for the AR backbone.
// If withScore is false in melody or full mode, the sequence stops at ABCStart,
// which is the prefix the score generation phase starts from.
func BuildPrompt(t *Tokenizer, cot Cot, style, lyrics string, abc []int32, withScore bool) []int32 {
	ids := []int32{EOD}
	ids = append(ids, t.Encode(requestText(cot, style, lyrics))...)
	ids = append(ids, ABCStart)
	if cot == CotOff {
		ids = append(ids, ABCEnd, MusicStart)
		return ids
	}
	if !withScore {
		return ids
	}
	ids = append(ids, abc...)
	ids = append(ids, ABCEnd, MusicStart)
	return ids
}

// BuildNegative assembles the unconditional token sequence for classifier-free guidance.
// In CotOff mode, the score slot is omitted entirely.
func BuildNegative(t *Tokenizer, cot Cot, abc []int32, withScore bool) []int32 {
	ids := []int32{EOD}
	ids = append(ids, t.Encode(instruction(cot))...)
	if cot == CotOff {
		ids = append(ids, MusicStart)
		return ids
	}
	ids = append(ids, ABCStart)
	if withScore && abc != nil {
		ids = append(ids, abc...)
	}
	ids = append(ids, ABCEnd, MusicStart)
	return ids
}
