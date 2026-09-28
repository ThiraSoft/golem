package laya

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"github.com/ThiraSoft/golem/nn"
)

// Named is a question with the identifier its answer is returned under.
type Named struct {
	ID string
	Question
}

// ParseQuestions reads a Jev request's questions, {"id": {"type": …}, …},
// keeping the order they were written in.
func ParseQuestions(raw json.RawMessage) ([]Named, error) {
	v, err := decodeOrdered(raw)
	if err != nil {
		return nil, err
	}
	o, ok := v.(*object)
	if !ok {
		return nil, fmt.Errorf("laya: the questions are an object from identifier to question")
	}
	// Decoding the whole object once more, by key, keeps each question's raw
	// bytes, which is what Question holds.
	var byID map[string]Question
	if err := json.Unmarshal(raw, &byID); err != nil {
		return nil, err
	}
	out := make([]Named, len(o.keys))
	for i, k := range o.keys {
		out[i] = Named{ID: k, Question: byID[k]}
	}
	return out, nil
}

// Answer is one question's answer, in the shape Jev and the checkpoint's own
// system_one return it. Probabilities are in the order of the options.
type Answer struct {
	ID    string
	Type  string
	Label string // a choice's pick
	// Score is a score's expected level; Noul a noul's probability of true.
	Score, Noul   float64
	Probabilities []Probability
	Legend        []string // a score's levels
	// Confidence is one minus the entropy of the distribution over the
	// options, normalized by its largest possible value.
	Confidence float64
	// Act is the act head's probability that the answer should be acted on.
	Act float64
}

type Probability struct {
	Label string
	P     float64
}

// Result is the answers to a request, in the order its questions came, and
// how many positions the pass read.
type Result struct {
	Answers     []Answer
	InputTokens int
}

// Decide answers every question about one state in one pass. The state is
// JSON: a string is read as the text it holds, anything else as Python's
// json.dumps would write it. It is safe to call from several goroutines; the
// passes take turns.
func (m *Model) Decide(ctx context.Context, state json.RawMessage, questions []Named) (*Result, error) {
	text, err := StateText(state)
	if err != nil {
		return nil, fmt.Errorf("laya: state: %w", err)
	}
	prepared := make([]*question, len(questions))
	seqs := make([]seq, len(questions))
	res := &Result{Answers: make([]Answer, len(questions))}
	for i, q := range questions {
		p, err := prepare(q.Question)
		if err != nil {
			return nil, fmt.Errorf("laya: question %q: %w", q.ID, err)
		}
		prepared[i] = p
	}
	// Every question's sequence carries the state, tokenized again for each
	// of them; thirty-two questions spent five milliseconds that way on one
	// core. The tokenizers are safe to share, so the questions are built on
	// the workers, side by side. On the workers and not on goroutines of
	// their own: those wait behind the workers, which are spinning between
	// two passes, and got no core until they stopped.
	nn.InParallel(len(questions), len(questions)<<20, func(first, last int) {
		for i := first; i < last; i++ {
			ids, markers := m.sequence(text, prepared[i])
			seqs[i] = seq{ids: ids, markers: markers, kind: prepared[i].kind}
		}
	})
	for i, q := range questions {
		if len(seqs[i].markers) != len(prepared[i].options) {
			return nil, fmt.Errorf("laya: question %q: its options do not fit in %d tokens", q.ID, m.Cfg.HeadMaxLen)
		}
		res.InputTokens += len(seqs[i].ids)
	}
	if len(seqs) == 0 {
		return res, nil
	}
	if err := m.acquire(ctx); err != nil {
		return nil, err
	}
	outs, err := m.forward(seqs)
	m.release()
	if err != nil {
		return nil, err
	}
	for i, q := range questions {
		res.Answers[i] = m.answer(q.ID, prepared[i], outs[i])
	}
	return res, nil
}

func (m *Model) acquire(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case m.turn <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *Model) release() { <-m.turn }

// answer calibrates one question's logits and reads them as its kind says.
func (m *Model) answer(id string, q *question, o output) Answer {
	k := len(o.logits)
	t := m.temperature(q.kind, k)
	z := make([]float32, k)
	for i, v := range o.logits {
		z[i] = v / t
	}
	p := softmax(z)
	a := Answer{ID: id, Type: kindNames[q.kind], Act: float64(o.act[0])}
	best := 0
	for i := range p {
		if p[i] > p[best] {
			best = i
		}
	}
	switch q.kind {
	case Choice, Score:
		for i, v := range p {
			a.Probabilities = append(a.Probabilities, Probability{q.labels[i], float64(v)})
		}
		a.Confidence = confidence(p)
		if q.kind == Choice {
			a.Label = q.labels[best]
		} else {
			for i, v := range p {
				a.Score += float64(i) * float64(v)
			}
			for _, opt := range q.options {
				_, level, _ := bytes.Cut([]byte(opt), []byte(": "))
				a.Legend = append(a.Legend, string(level))
			}
		}
	case Noul:
		a.Noul = float64(p[1])
	}
	return a
}

// temperature is the one fitted for this kind and number of options, or the
// kind's own when none was.
func (m *Model) temperature(kind, k int) float32 {
	size := "11+"
	switch {
	case k <= 2:
		size = "2"
	case k <= 5:
		size = "3-5"
	case k <= 10:
		size = "6-10"
	}
	if t, ok := m.Cfg.TemperatureBy[kindNames[kind]+":"+size]; ok {
		return t
	}
	return m.Cfg.Temperature[kind]
}

func confidence(p []float32) float64 {
	if len(p) < 2 {
		return 1
	}
	var ent float64
	for _, v := range p {
		ent -= float64(v) * math.Log(math.Min(math.Max(float64(v), 1e-12), 1))
	}
	return 1 - ent/math.Log(float64(len(p)))
}

// round4 is round(v, 4), which is what the reference prints.
func round4(v float64) json.Number {
	return json.Number(strconv.FormatFloat(math.Round(v*1e4)/1e4, 'f', -1, 64))
}

// MarshalJSON writes the answer as system_one does, rounded to four places.
func (a Answer) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	field := func(name string, v any) {
		if b.Len() > 1 {
			b.WriteByte(',')
		}
		k, _ := json.Marshal(name)
		val, _ := json.Marshal(v)
		b.Write(k)
		b.WriteByte(':')
		b.Write(val)
	}
	probs := func() {
		if b.Len() > 1 {
			b.WriteByte(',')
		}
		b.WriteString(`"probabilities":{`)
		for i, p := range a.Probabilities {
			if i > 0 {
				b.WriteByte(',')
			}
			k, _ := json.Marshal(p.Label)
			b.Write(k)
			b.WriteByte(':')
			b.WriteString(string(round4(p.P)))
		}
		b.WriteByte('}')
	}
	b.WriteByte('{')
	field("type", a.Type)
	switch a.Type {
	case "choice":
		field("choice", a.Label)
		probs()
		field("confidence", round4(a.Confidence))
	case "score":
		field("score", round4(a.Score))
		b.WriteString(`,"legend":{`)
		for i, l := range a.Legend {
			if i > 0 {
				b.WriteByte(',')
			}
			k, _ := json.Marshal(strconv.Itoa(i))
			v, _ := json.Marshal(l)
			b.Write(k)
			b.WriteByte(':')
			b.Write(v)
		}
		b.WriteByte('}')
		probs()
		field("confidence", round4(a.Confidence))
	case "noul":
		field("noul", round4(a.Noul))
	}
	field("rl_agent", map[string]json.Number{"act_probability": round4(a.Act)})
	b.WriteByte('}')
	return b.Bytes(), nil
}

// MarshalJSON writes the result as system_one does: the answers under their
// identifiers, in order, and the usage.
func (r *Result) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteString(`{"model":"laya","answers":{`)
	for i, a := range r.Answers {
		if i > 0 {
			b.WriteByte(',')
		}
		k, _ := json.Marshal(a.ID)
		v, err := a.MarshalJSON()
		if err != nil {
			return nil, err
		}
		b.Write(k)
		b.WriteByte(':')
		b.Write(v)
	}
	fmt.Fprintf(&b, `},"usage":{"input_tokens":%d,"output_tokens":0}}`, r.InputTokens)
	return b.Bytes(), nil
}
