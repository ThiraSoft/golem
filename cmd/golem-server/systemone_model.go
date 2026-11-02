package main

// POST /v1/systemone, answered by the chat model when no Laya was given.
//
// The request is Jev's, with two things a chat model can use and Laya cannot:
// a "system" turn, and a state that may be an array of chat completions'
// content parts, text and pictures, in the order the model should read them.
// A client that keeps what does not change in front (a persona, rules, a
// history) and what does at the end (a screenshot, the moment) has the front
// read once, as a conversation's prefix is: weigh.go rewinds the cache to the
// end of the state between questions, and the next request finds the same
// beginning in the same slot.
//
// Each question is put to the model as a turn of its own after the state, and
// its options are weighed rather than drawn (weigh.go). A choice's options are
// its labels, a score's are its levels' numbers, a noul's are Yes and No. The
// answer is Laya's shape, with the distribution over the options for
// probabilities and no act head.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/engine"
	"github.com/ThiraSoft/golem/laya"
)

// sentinel stands for the turn's content while the template renders, so that
// what goes before it and after it can be cut out and the content written by
// hand: the template puts a turn's pictures before its text, and a state has
// to keep its own order.
const sentinel = "GOLEMSYSTEMONECONTENT"

// A requestError is the client's: a state or a question the model cannot be
// asked as written.
type requestError struct{ error }

func (s *Server) systemoneByModel(w http.ResponseWriter, r *http.Request, system string,
	state json.RawMessage, questions []laya.Named) {
	res, err := s.weighQuestions(w, r, system, state, questions)
	if err != nil {
		var bad requestError
		switch {
		case errors.As(err, &bad):
			detail(w, http.StatusUnprocessableEntity, err.Error())
		case r.Context().Err() != nil:
		default:
			detail(w, http.StatusInternalServerError, err.Error())
		}
		return
	}
	if rec, ok := w.(*recorder); ok {
		plural := "s"
		if len(questions) == 1 {
			plural = ""
		}
		rec.note = fmt.Sprintf("%d question%s, %d positions fed", len(questions), plural, res.InputTokens)
	}
	writeJSON(w, http.StatusOK, res)
}

// asked is one question made into words for the model, and its options.
type asked struct {
	prepared *laya.Prepared
	text     string
	options  []string
}

func (s *Server) weighQuestions(w http.ResponseWriter, r *http.Request, system string, state json.RawMessage,
	questions []laya.Named) (*laya.Result, error) {
	body, pictures, err := s.stateBody(state)
	if err != nil {
		return nil, err
	}
	head, tail, err := s.turn(system)
	if err != nil {
		return nil, err
	}
	closing, err := s.closing()
	if err != nil {
		return nil, err
	}
	qs := make([]asked, len(questions))
	for i, q := range questions {
		p, err := laya.Prepare(q.Question)
		if err != nil {
			return nil, requestError{fmt.Errorf("question %q: %w", q.ID, err)}
		}
		qs[i] = askedOf(p)
	}

	var rows [][][]float32
	for _, raw := range pictures {
		r, err := s.images.Encode(raw, s.vision.EncodeImage)
		if err != nil {
			return nil, requestError{err}
		}
		rows = append(rows, r)
	}
	build := func(ids []int32) (engine.Prompt, error) {
		if rows == nil {
			return engine.TextPrompt(ids), nil
		}
		p, err := s.vision.Prompt(ids, rows, nil)
		if err != nil {
			return nil, requestError{err}
		}
		return p, nil
	}

	// The state is what every question starts with. Its tokens are whatever
	// all of them agree on, so that a merge across where the state ends is
	// never taken for a different state.
	//
	// Every one of these texts is the whole state and more, and there are as
	// many of them as options: 42 ms one after the other for two questions of
	// six and five options on a state of about 2600 positions. Encode is safe
	// to call from several goroutines, so they go side by side.
	texts := []string{head + body}
	for _, q := range qs {
		prompt := head + body + "\n\n" + q.text + tail
		texts = append(texts, prompt)
		for _, o := range q.options {
			texts = append(texts, prompt+o)
		}
	}
	encoded := make([][]int32, len(texts))
	var wg sync.WaitGroup
	for i, text := range texts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			encoded[i] = s.vocab.Encode(text, false, true)
		}()
	}
	wg.Wait()
	stateIDs, encoded := encoded[0], encoded[1:]
	ws := make([]weighing, len(qs))
	whole := make([][]int32, len(qs))
	for i, q := range qs {
		whole[i], encoded = encoded[0], encoded[1:]
		stateIDs = stateIDs[:commonPrefix(stateIDs, whole[i])]
		for _, o := range q.options {
			var ids []int32
			ids, encoded = encoded[0], encoded[1:]
			if commonPrefix(ids, whole[i]) != len(whole[i]) || len(ids) == len(whole[i]) {
				return nil, requestError{fmt.Errorf("question %q: the option %q does not follow the question as tokens of its own", questions[i].ID, o)}
			}
			ws[i].options = append(ws[i].options, slices.Concat(ids[len(whole[i]):], closing))
		}
	}
	for i := range ws {
		ws[i].tail = whole[i][len(stateIDs):]
	}

	front, err := build(stateIDs)
	if err != nil {
		return nil, err
	}
	sl, waited, err := s.pool.Acquire(r.Context(), front.Tokens())
	if err != nil {
		return nil, err
	}
	defer s.pool.Release(sl)
	waitedFor(w, waited, sl.index)
	sl.ctx.runner.Enter()
	defer sl.ctx.runner.Leave()

	lps, fed, err := sl.ctx.Weigh(r.Context(), build, stateIDs, ws, s.emptyThought(), sl.gen.Width())
	if err != nil {
		if r.Context().Err() != nil {
			return nil, err
		}
		return nil, requestError{err}
	}
	res := &laya.Result{Model: s.name, InputTokens: fed}
	for i, q := range questions {
		res.Answers = append(res.Answers, laya.Answered(q.ID, qs[i].prepared, distribution(lps[i])))
	}
	return res, nil
}

// distribution is the softmax of the options' log-probabilities, in the order
// laya.Answered reads: a noul's false, then true.
func distribution(lp []float64) []float64 {
	m := math.Inf(-1)
	for _, v := range lp {
		m = math.Max(m, v)
	}
	p := make([]float64, len(lp))
	var sum float64
	for i, v := range lp {
		p[i] = math.Exp(v - m)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}

// askedOf writes a question for the model, and says what its options are.
func askedOf(p *laya.Prepared) asked {
	var b strings.Builder
	q := asked{prepared: p}
	switch p.Kind {
	case "choice":
		b.WriteString(p.Instructions + "\n\nOptions:\n")
		for _, o := range p.Options {
			b.WriteString("- " + o + "\n")
		}
		b.WriteString("\nAnswer with the name of one option, exactly as written, and nothing else.")
		q.options = p.Labels
	case "score":
		b.WriteString(p.Instructions + "\n\nLevels:\n")
		for i, o := range p.Options {
			_, level, _ := strings.Cut(o, ": ")
			b.WriteString(strconv.Itoa(i) + ": " + level + "\n")
		}
		b.WriteString("\nAnswer with the number of one level and nothing else.")
		q.options = p.Labels
	case "noul":
		_, no, _ := strings.Cut(p.Options[0], ": ")
		_, yes, _ := strings.Cut(p.Options[1], ": ")
		b.WriteString("Statement: " + p.Instructions + "\n\nYes: " + yes + "\nNo: " + no +
			"\n\nDoes the statement hold? Answer Yes or No and nothing else.")
		q.options = []string{"No", "Yes"}
	}
	q.text = b.String()
	return q
}

// stateBody writes the state as the model reads it, a marker standing where
// each picture goes, and returns the pictures in that order.
func (s *Server) stateBody(state json.RawMessage) (string, [][]byte, error) {
	parts, ok := contentParts(state)
	if !ok {
		text, err := laya.StateText(state)
		if err != nil {
			return "", nil, requestError{fmt.Errorf("state: %w", err)}
		}
		return text, nil, nil
	}
	var b strings.Builder
	var pictures [][]byte
	for i, p := range parts {
		if i > 0 {
			b.WriteString("\n")
		}
		switch p.Type {
		case "text":
			b.WriteString(p.Text)
		case "image_url":
			if s.vision == nil || !s.vision.CanSee() {
				return "", nil, requestError{fmt.Errorf("this model was started without a projector that can see, so it cannot look at a picture: start the server with -mmproj")}
			}
			data, err := readImageURL(p.ImageURL.URL)
			if err != nil {
				return "", nil, requestError{fmt.Errorf("state part %d: %w", i, err)}
			}
			marker, err := s.pictureMarker()
			if err != nil {
				return "", nil, err
			}
			b.WriteString(marker)
			pictures = append(pictures, data)
		default:
			return "", nil, requestError{fmt.Errorf("state part %d is of type %q, and a state holds text and image_url", i, p.Type)}
		}
	}
	return b.String(), pictures, nil
}

// contentParts reads a state that is an array of content parts. Any other
// state, an array of something else among them, is Jev's.
func contentParts(state json.RawMessage) ([]contentPart, bool) {
	if t := bytes.TrimSpace(state); len(t) == 0 || t[0] != '[' {
		return nil, false
	}
	var raw []map[string]json.RawMessage
	if json.Unmarshal(state, &raw) != nil || len(raw) == 0 {
		return nil, false
	}
	for _, r := range raw {
		if _, ok := r["type"]; !ok {
			return nil, false
		}
	}
	var parts []contentPart
	if json.Unmarshal(state, &parts) != nil {
		return nil, false
	}
	return parts, true
}

// turn is the template's text before a user turn's content and after it,
// generation prompt included and thinking off, with the system turn in front
// when there is one.
func (s *Server) turn(system string) (head, tail string, err error) {
	var msgs []chat.Message
	if system != "" {
		msgs = append(msgs, chat.Message{Role: "system", Content: system})
	}
	msgs = append(msgs, chat.Message{Role: "user", Content: sentinel})
	text, err := s.tpl.Render(msgs, chat.Options{AddGenerationPrompt: true})
	if err != nil {
		return "", "", err
	}
	head, tail, ok := strings.Cut(text, sentinel)
	if !ok {
		return "", "", fmt.Errorf("the chat template did not write the turn's content where it could be found")
	}
	return head, tail, nil
}

// emptyThought is a thought opened and closed with nothing in it, as Gemma 4
// writes one before an answer it was not asked to think about; nil for a
// template with no reasoning markers. Weigh puts it before the options only
// when the model reaches for it.
func (s *Server) emptyThought() []int32 {
	open, close := s.tpl.ReasoningMarkers()
	if open == "" {
		return nil
	}
	return s.vocab.Encode(open+"\n"+close, false, true)
}

// closing is the tokens that end the model's turn, which every option is
// weighed with: without them "heal" would be weighed as every answer that
// starts with it, "heal the tank" among them. It is read off the template, as
// the end of an answer, through the first token that ends a generation, and
// is empty for a template that writes none.
func (s *Server) closing() ([]int32, error) {
	text, err := s.tpl.Render([]chat.Message{{Role: "user", Content: sentinel},
		{Role: "assistant", Content: sentinel + "A"}}, chat.Options{})
	if err != nil {
		return nil, err
	}
	_, after, ok := strings.Cut(text, sentinel+"A")
	if !ok {
		return nil, fmt.Errorf("the chat template did not write the answer where it could be found")
	}
	ids := s.vocab.Encode(after, false, true)
	for i, id := range ids {
		if s.vocab.IsEOG(id) {
			return ids[:i+1], nil
		}
	}
	return nil, nil
}

// pictureMarker is what the template writes for one picture, which the
// projector's Prompt finds and fills. It is read off the template once, by
// rendering a turn with a picture and one without.
func (s *Server) pictureMarker() (string, error) {
	s.markerOnce.Do(func() {
		render := func(images [][]byte) (string, error) {
			return s.tpl.Render([]chat.Message{{Role: "user", Content: sentinel, Images: images}},
				chat.Options{AddGenerationPrompt: true})
		}
		plain, err := render(nil)
		if err != nil {
			s.markerErr = err
			return
		}
		seen, err := render([][]byte{{0}})
		if err != nil {
			s.markerErr = err
			return
		}
		before, _, _ := strings.Cut(seen, sentinel)
		plainBefore, _, _ := strings.Cut(plain, sentinel)
		n := 0
		for n < len(before) && n < len(plainBefore) && before[n] == plainBefore[n] {
			n++
		}
		s.marker = before[n:]
		if s.marker == "" {
			s.markerErr = fmt.Errorf("the chat template writes nothing for a picture")
		}
	})
	return s.marker, s.markerErr
}

// markerState is the picture marker, read once. Server carries it.
type markerState struct {
	markerOnce sync.Once
	marker     string
	markerErr  error
}
