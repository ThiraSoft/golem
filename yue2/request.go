package yue2

// A request, as yue2.cpp's src/request.h and request.cpp hold it: the same
// fields under the same JSON names with the same defaults, so a request or a
// replay written by yue2.cpp or musicbench reads here as it is. A field that
// is missing, or of the wrong JSON type, keeps its default, as yyjson's
// typed getters leave it.

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// The output formats a request names.
const (
	FormatMP3   = "mp3"
	FormatWAV16 = "wav16"
	FormatWAV24 = "wav24"
	FormatWAV32 = "wav32"
)

// Adapter is one adapter a request stacks on the backbone, by name.
type Adapter struct {
	Name  string
	Scale float32
}

// Request is one song to render. The comments give yue2.cpp's defaults,
// which DefaultRequest fills.
type Request struct {
	Style, Lyrics string // ""

	// ABC is the score; empty in melody or full mode makes the model write one.
	ABC string // ""
	// Cot is "full", "melody" or "off".
	Cot string // "full"

	// Duration in seconds caps the semantic stage's budget.
	Duration float32 // 360

	// LMSeed is the Philox key of the token draws, Seed the mt19937 seed of
	// the acoustic noise; -1 draws one.
	LMSeed, Seed int64 // -1
	Steps        int   // 32

	LMBatchSize, SynthBatchSize int // 1

	ABCSampling, SemanticSampling Sampling // the checkpoint's presets

	// SemanticTokens, a CSV of codec values, replaces the semantic stage.
	SemanticTokens string // ""

	// CfgScale below zero is the protocol's default guidance.
	CfgScale float32 // -1

	// PeakClip is the normalization's percentile control, 1 - PeakClip/1e6.
	PeakClip int // 10

	OutputFormat string // "mp3"
	MP3Bitrate   int    // 128

	Adapters []Adapter

	// SourceTokens, SourceStart and SourceEnd are a continuation's source.
	SourceTokens string  // ""
	SourceStart  float32 // 0
	SourceEnd    float32 // -1
}

// DefaultRequest is request_init's request.
func DefaultRequest() Request {
	return Request{
		Cot:              "full",
		Duration:         360,
		LMSeed:           -1,
		Seed:             -1,
		Steps:            32,
		LMBatchSize:      1,
		SynthBatchSize:   1,
		ABCSampling:      ABCSampling,
		SemanticSampling: SemanticSampling,
		CfgScale:         -1,
		PeakClip:         10,
		OutputFormat:     FormatMP3,
		MP3Bitrate:       128,
		SourceEnd:        -1,
	}
}

// ParseRequest reads a request's JSON over the defaults.
func ParseRequest(data []byte) (Request, error) {
	r := DefaultRequest()
	var root map[string]json.RawMessage
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	if err := d.Decode(&root); err != nil {
		return r, fmt.Errorf("yue2: request: %w", err)
	}
	if root == nil {
		return r, fmt.Errorf("yue2: request: the root is not an object")
	}
	o := fields(root)
	o.str("style", &r.Style)
	o.str("lyrics", &r.Lyrics)
	o.str("abc", &r.ABC)
	o.str("cot", &r.Cot)
	o.f32("duration", &r.Duration)
	o.i64("lm_seed", &r.LMSeed)
	o.i64("seed", &r.Seed)
	o.int("peak_clip", &r.PeakClip)
	o.int("steps", &r.Steps)
	o.int("lm_batch_size", &r.LMBatchSize)
	o.int("synth_batch_size", &r.SynthBatchSize)
	o.sampling("abc_sampling", &r.ABCSampling)
	o.sampling("semantic_sampling", &r.SemanticSampling)
	o.str("semantic_tokens", &r.SemanticTokens)
	o.f32("cfg_scale", &r.CfgScale)
	o.str("output_format", &r.OutputFormat)
	o.int("mp3_bitrate", &r.MP3Bitrate)
	o.str("source_tokens", &r.SourceTokens)
	o.f32("source_start", &r.SourceStart)
	o.f32("source_end", &r.SourceEnd)
	if raw, ok := root["adapters"]; ok {
		var list []map[string]json.RawMessage
		if json.Unmarshal(raw, &list) == nil {
			for _, item := range list {
				a := Adapter{Scale: 1}
				f := fields(item)
				if f.str("name", &a.Name) {
					f.f32("scale", &a.Scale)
					r.Adapters = append(r.Adapters, a)
				}
			}
		}
	}
	return r, nil
}

// fields reads one JSON object's members by type, as yyjson's getters do: a
// member of another type is left alone.
type fields map[string]json.RawMessage

func (o fields) str(key string, dst *string) bool {
	raw, ok := o[key]
	if !ok {
		return false
	}
	// json.Unmarshal takes null into a string as nothing at all, where
	// yyjson_is_str refuses it: a null member keeps its default.
	var s *string
	if json.Unmarshal(raw, &s) != nil || s == nil {
		return false
	}
	*dst = *s
	return true
}

// number is a member's JSON number text, and whether it is an integer in
// yyjson's sense: no fraction and no exponent.
func (o fields) number(key string) (string, bool, bool) {
	raw, ok := o[key]
	if !ok {
		return "", false, false
	}
	s := strings.TrimSpace(string(raw))
	if _, err := strconv.ParseFloat(s, 64); err != nil {
		return "", false, false
	}
	return s, !strings.ContainsAny(s, ".eE"), true
}

func (o fields) f32(key string, dst *float32) {
	if s, _, ok := o.number(key); ok {
		v, _ := strconv.ParseFloat(s, 64)
		*dst = float32(v)
	}
}

// i64 reads an integer member exactly: a seed past 2^53 would not survive a
// float64. An integer past int64 is yyjson's uint64, and wraps as its
// yyjson_get_sint does.
func (o fields) i64(key string, dst *int64) {
	s, integer, ok := o.number(key)
	if !ok || !integer {
		return
	}
	if v, err := strconv.ParseInt(s, 10, 64); err == nil {
		*dst = v
	} else if u, err := strconv.ParseUint(s, 10, 64); err == nil {
		*dst = int64(u)
	}
}

func (o fields) int(key string, dst *int) {
	var v int64
	set := false
	if s, integer, ok := o.number(key); ok && integer {
		if x, err := strconv.ParseInt(s, 10, 64); err == nil {
			v, set = x, true
		}
	}
	if set {
		*dst = int(int32(v))
	}
}

func (o fields) sampling(key string, s *Sampling) {
	raw, ok := o[key]
	if !ok {
		return
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(raw, &m) != nil || m == nil {
		return
	}
	f := fields(m)
	f.f32("temperature", &s.Temperature)
	f.f32("top_p", &s.TopP)
	f.int("top_k", &s.TopK)
	f.f32("repetition_penalty", &s.RepetitionPenalty)
	f.int("penalty_window", &s.PenaltyWindow)
	f.int("min_tokens", &s.MinTokens)
	f.int("max_tokens", &s.MaxTokens)
}

// JSON is request_to_json with sparse set: only the fields away from their
// default, in yue2.cpp's order, floats in the shortest text that reads back
// to the same float32.
func (r Request) JSON() []byte {
	d := DefaultRequest()
	var b jsonObject
	if r.Style != d.Style {
		b.add("style", r.Style)
	}
	if r.Lyrics != d.Lyrics {
		b.add("lyrics", r.Lyrics)
	}
	if r.ABC != d.ABC {
		b.add("abc", r.ABC)
	}
	if r.Cot != d.Cot {
		b.add("cot", r.Cot)
	}
	if r.Duration != d.Duration {
		b.add("duration", r.Duration)
	}
	if r.LMSeed != d.LMSeed {
		b.add("lm_seed", r.LMSeed)
	}
	if r.Seed != d.Seed {
		b.add("seed", r.Seed)
	}
	if r.Steps != d.Steps {
		b.add("steps", r.Steps)
	}
	if r.LMBatchSize != d.LMBatchSize {
		b.add("lm_batch_size", r.LMBatchSize)
	}
	if r.SynthBatchSize != d.SynthBatchSize {
		b.add("synth_batch_size", r.SynthBatchSize)
	}
	addSampling(&b, "abc_sampling", r.ABCSampling, d.ABCSampling)
	addSampling(&b, "semantic_sampling", r.SemanticSampling, d.SemanticSampling)
	if r.SemanticTokens != d.SemanticTokens {
		b.add("semantic_tokens", r.SemanticTokens)
	}
	if r.CfgScale != d.CfgScale {
		b.add("cfg_scale", r.CfgScale)
	}
	if r.OutputFormat != d.OutputFormat {
		b.add("output_format", r.OutputFormat)
	}
	if r.PeakClip != d.PeakClip {
		b.add("peak_clip", r.PeakClip)
	}
	if r.MP3Bitrate != d.MP3Bitrate {
		b.add("mp3_bitrate", r.MP3Bitrate)
	}
	if r.SourceTokens != d.SourceTokens {
		b.add("source_tokens", r.SourceTokens)
	}
	if r.SourceStart != d.SourceStart {
		b.add("source_start", r.SourceStart)
	}
	if r.SourceEnd != d.SourceEnd {
		b.add("source_end", r.SourceEnd)
	}
	if len(r.Adapters) > 0 {
		var list []string
		for _, a := range r.Adapters {
			var item jsonObject
			item.add("name", a.Name)
			item.add("scale", a.Scale)
			list = append(list, item.text("    "))
		}
		b.raw("adapters", "[\n    "+strings.Join(list, ",\n    ")+"\n  ]")
	}
	return []byte(b.text("") + "\n")
}

func addSampling(b *jsonObject, key string, s, d Sampling) {
	var o jsonObject
	if s.Temperature != d.Temperature {
		o.add("temperature", s.Temperature)
	}
	if s.TopP != d.TopP {
		o.add("top_p", s.TopP)
	}
	if s.TopK != d.TopK {
		o.add("top_k", s.TopK)
	}
	if s.RepetitionPenalty != d.RepetitionPenalty {
		o.add("repetition_penalty", s.RepetitionPenalty)
	}
	if s.PenaltyWindow != d.PenaltyWindow {
		o.add("penalty_window", s.PenaltyWindow)
	}
	if s.MinTokens != d.MinTokens {
		o.add("min_tokens", s.MinTokens)
	}
	if s.MaxTokens != d.MaxTokens {
		o.add("max_tokens", s.MaxTokens)
	}
	if len(o.members) > 0 {
		b.raw(key, o.text("  "))
	}
}

// jsonObject writes an object's members in the order they come, two spaces
// an indent level, as yyjson's pretty writer does.
type jsonObject struct{ members []string }

func (o *jsonObject) raw(key, value string) {
	k, _ := json.Marshal(key)
	o.members = append(o.members, string(k)+": "+value)
}

func (o *jsonObject) add(key string, v any) {
	var text string
	switch v := v.(type) {
	case string:
		var b bytes.Buffer
		e := json.NewEncoder(&b)
		e.SetEscapeHTML(false)
		e.Encode(v)
		text = strings.TrimSuffix(b.String(), "\n")
	case float32:
		text = strconv.FormatFloat(float64(v), 'g', -1, 32)
		if !strings.ContainsAny(text, ".eEn") {
			text += ".0"
		}
	case int:
		text = strconv.Itoa(v)
	case int64:
		text = strconv.FormatInt(v, 10)
	}
	o.raw(key, text)
}

func (o *jsonObject) text(indent string) string {
	if len(o.members) == 0 {
		return "{}"
	}
	return "{\n" + indent + "  " + strings.Join(o.members, ",\n"+indent+"  ") + "\n" + indent + "}"
}

// randomSeed is request.cpp's random_seed: 64 random bits shifted right
// once, so the seed is never negative.
func randomSeed() int64 {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(fmt.Sprintf("yue2: no random seed: %v", err))
	}
	return int64(binary.LittleEndian.Uint64(b[:]) >> 1)
}

// Resolve is request_resolve_seed: a negative seed becomes a random one.
func (r Request) Resolve() Request {
	if r.LMSeed < 0 {
		r.LMSeed = randomSeed()
	}
	if r.Seed < 0 {
		r.Seed = randomSeed()
	}
	return r
}

// ParseCodes reads a semantic stream's CSV, as pipeline_parse_tokens: codec
// values in [0, CodecSize), separated by commas or white space.
func ParseCodes(csv string) ([]int32, error) {
	var out []int32
	for _, f := range strings.FieldsFunc(csv, func(c rune) bool {
		return c == ',' || c == ' ' || c == '\n' || c == '\r' || c == '\t'
	}) {
		v, err := strconv.ParseInt(f, 10, 64)
		if err != nil || v < 0 || v >= CodecSize {
			return nil, fmt.Errorf("yue2: semantic token %q outside [0, %d)", f, CodecSize)
		}
		out = append(out, int32(v))
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("yue2: an empty semantic stream")
	}
	return out, nil
}

// FormatCodes writes a semantic stream's CSV.
func FormatCodes(codes []int32) string {
	var b strings.Builder
	for i, c := range codes {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(int(c)))
	}
	return b.String()
}
