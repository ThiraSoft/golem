package main

import (
	"encoding/json"
	"math"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/chat"
)

// An engine with a taste: whatever it has read, it would rather write some
// words than others, by as much as likes says. It records what it was fed.
type tasteEngine struct {
	recordingEngine
	vocab *wordVocab
	likes map[string]float32
}

func (e *tasteEngine) LogitsBatch(hidden [][]float32, out [][]float32) {
	for _, o := range out {
		clear(o)
		for w, v := range e.likes {
			o[e.vocab.id(w)] = v
		}
	}
}

func (e *tasteEngine) Logits(hidden []float32, out []float32) {
	e.LogitsBatch([][]float32{hidden}, [][]float32{out})
}

func newTasteServer(t *testing.T, likes map[string]float32) (*Server, *tasteEngine) {
	t.Helper()
	v := newWordVocab()
	e := &tasteEngine{vocab: v, likes: likes}
	for w := range likes {
		v.id(w)
	}
	ctx := NewContext(running(t, e), 0, 4096, time.Now, 0)
	g := NewGenerator(ctx, v, wordTemplate{}, 4096, 16)
	return NewServer(poolOf(g), v, "test-model", wordTemplate{}, greedy()), e
}

func ask(t *testing.T, s *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body)))
	return rec
}

// What the model read, as words.
func (e *tasteEngine) read() string {
	var words []string
	for _, id := range e.fed {
		words = append(words, e.vocab.Piece(id, false))
	}
	return strings.Join(words, " ")
}

// Without -laya the chat model answers, in Laya's shape: every option with
// its probability, which is the softmax of what the model would have given
// each option, and no act head.
func TestSystemOneAnswersWithTheChatModel(t *testing.T) {
	s, e := newTasteServer(t, map[string]float32{"ATTACK": 2, "HOLD": 1})
	rec := ask(t, s, `{"state":{"hp":40},"questions":{
		"order":{"type":"choice","instructions":"What now?","criteria":{"HOLD":"stay put","ATTACK":"go","REGROUP":""}}}}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var out struct {
		Model   string `json:"model"`
		Answers map[string]struct {
			Choice        string             `json:"choice"`
			Probabilities map[string]float64 `json:"probabilities"`
			Confidence    float64            `json:"confidence"`
			Agent         *json.RawMessage   `json:"rl_agent"`
		} `json:"answers"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	a := out.Answers["order"]
	if out.Model != "test-model" || a.Choice != "ATTACK" || a.Agent != nil {
		t.Fatalf("%s", rec.Body)
	}
	z := math.Exp(2) + math.Exp(1) + 1
	for label, want := range map[string]float64{"ATTACK": math.Exp(2) / z, "HOLD": math.Exp(1) / z, "REGROUP": 1 / z} {
		if math.Abs(a.Probabilities[label]-want) > 1e-4 {
			t.Fatalf("%s is %v, want %v: %s", label, a.Probabilities[label], want, rec.Body)
		}
	}
	read := e.read()
	for _, want := range []string{`{"hp":`, "What", "HOLD:", "stay", "ATTACK:", "REGROUP"} {
		if !strings.Contains(read, want) {
			t.Fatalf("the model read %q, without %q", read, want)
		}
	}
}

// A noul is the probability of Yes against No.
func TestSystemOneNoulIsYesAgainstNo(t *testing.T) {
	s, _ := newTasteServer(t, map[string]float32{"Yes": 1, "No": -1})
	rec := ask(t, s, `{"state":"the boss is low","questions":{"low":{"type":"noul","instructions":"The boss is nearly dead."}}}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var out struct {
		Answers map[string]struct {
			Noul float64 `json:"noul"`
		} `json:"answers"`
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	want := math.Exp(1) / (math.Exp(1) + math.Exp(-1))
	if math.Abs(out.Answers["low"].Noul-want) > 1e-4 {
		t.Fatalf("noul %v, want %v", out.Answers["low"].Noul, want)
	}
}

// A score is the expected level, read off the levels' numbers.
func TestSystemOneScoreIsTheExpectedLevel(t *testing.T) {
	s, _ := newTasteServer(t, map[string]float32{"2": 10})
	rec := ask(t, s, `{"state":"x","questions":{"danger":{"type":"score","instructions":"How bad?","criteria":["none","some","a lot"]}}}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var out struct {
		Answers map[string]struct {
			Score  float64           `json:"score"`
			Legend map[string]string `json:"legend"`
		} `json:"answers"`
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	if a := out.Answers["danger"]; a.Score < 1.99 || a.Legend["2"] != "a lot" {
		t.Fatalf("%s", rec.Body)
	}
}

// The system prompt and the parts of a state reach the model in order, a
// stable part before a changing one.
func TestSystemOneReadsTheSystemAndTheParts(t *testing.T) {
	s, e := newTasteServer(t, map[string]float32{"Yes": 1})
	rec := ask(t, s, `{"system":"You are Nova.","state":[{"type":"text","text":"rules first"},{"type":"text","text":"then now"}],
		"questions":{"q":{"type":"noul","instructions":"Fine."}}}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	read := e.read()
	nova, rules, now := strings.Index(read, "Nova."), strings.Index(read, "rules"), strings.Index(read, "now")
	if nova < 0 || rules < nova || now < rules {
		t.Fatalf("the model read %q", read)
	}
}

// A picture with no projector to see it is refused by name.
func TestSystemOneRefusesAPictureWithoutAProjector(t *testing.T) {
	s, _ := newTasteServer(t, map[string]float32{"Yes": 1})
	rec := ask(t, s, `{"state":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}],
		"questions":{"q":{"type":"noul","instructions":"Fine."}}}`)
	if rec.Code != 422 || !strings.Contains(rec.Body.String(), "-mmproj") {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}

// A question the model cannot be asked is refused the way Laya refuses it.
func TestSystemOneRefusesAQuestionOfNoKind(t *testing.T) {
	s, _ := newTasteServer(t, map[string]float32{"Yes": 1})
	rec := ask(t, s, `{"state":"x","questions":{"q":{"type":"guess","instructions":"?"}}}`)
	if rec.Code != 422 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}

// A template that closes every turn, as the real ones do.
type closingTemplate struct{ wordTemplate }

func (closingTemplate) Render(msgs []chat.Message, opt chat.Options) (string, error) {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.Role + " " + m.Content + " <turn|> ")
	}
	if opt.AddGenerationPrompt {
		b.WriteString("assistant ")
	}
	return b.String(), nil
}

// An engine whose taste depends on the word it read last.
type followEngine struct {
	recordingEngine
	vocab *wordVocab
	after map[string]map[string]float32
}

func (e *followEngine) LogitsBatch(hidden [][]float32, out [][]float32) {
	for i, o := range out {
		clear(o)
		likes, ok := e.after[e.vocab.Piece(int32(hidden[i][0]), false)]
		if !ok {
			likes = e.after[""]
		}
		for w, v := range likes {
			o[e.vocab.id(w)] = v
		}
	}
}

func (e *followEngine) Logits(hidden []float32, out []float32) {
	e.LogitsBatch([][]float32{hidden}, [][]float32{out})
}

// An option is weighed through the end of the answer: "heal" is the answer
// that stops there, not every answer that starts with it.
func TestSystemOneWeighsAnOptionWithItsEnd(t *testing.T) {
	v := newWordVocab()
	e := &followEngine{vocab: v, after: map[string]map[string]float32{
		"":     {"heal": 20},
		"heal": {"the": 20},
		"the":  {"tank": 20},
		"tank": {"<turn|>": 20},
	}}
	ctx := NewContext(running(t, e), 0, 4096, time.Now, 0)
	s := NewServer(poolOf(NewGenerator(ctx, v, closingTemplate{}, 4096, 16)), v, "m", closingTemplate{}, greedy())
	rec := ask(t, s, `{"state":"x","questions":{"g":{"type":"choice","instructions":"Which?","criteria":["heal","heal the tank"]}}}`)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"choice":"heal the tank"`) {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}
