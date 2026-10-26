package main

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/engine"
)

// liveDecider is a server over a real model and its projector, answering
// /v1/systemone with the chat model.
func liveDecider(t *testing.T) *Server {
	t.Helper()
	path, proj := os.Getenv("GOLEM_MODEL"), os.Getenv("GOLEM_MMPROJ")
	if path == "" || proj == "" {
		t.Skip("GOLEM_MODEL and GOLEM_MMPROJ are not both set")
	}
	m, err := engine.Open(path, 4096)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	if err := m.OpenProjector(proj); err != nil {
		t.Fatal(err)
	}
	r := running(t, m.Forward)
	v, _ := m.Media()
	r.SetVision(v)
	if b, ok := m.Branches(); ok {
		r.SetBranches(b)
	}
	ctx := NewContext(r, m.Window, 4096, time.Now, 0)
	ctx.SetRing(m.WindowRing())
	s := NewServer(poolOf(NewGenerator(ctx, m.Vocab, m.Template, m.Vocabulary, 16)),
		m.Vocab, "live", m.Template, m.Sampling)
	s.SetVision(v)
	return s
}

// A picture and two questions about it: the model sees what is in it, and
// asked again it reads only the questions.
func TestLiveSystemOneLooksAtAPicture(t *testing.T) {
	s := liveDecider(t)
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "gemma", "shapes.png"))
	if err != nil {
		t.Skipf("the test image is not on this machine: %v", err)
	}
	body := `{"system":"You look at pictures and answer exactly.",
		"state":[{"type":"text","text":"Here is the picture."},
		         {"type":"image_url","image_url":{"url":"data:image/png;base64,` + base64.StdEncoding.EncodeToString(raw) + `"}}],
		"questions":{
		  "shape":{"type":"choice","instructions":"Which shape is in the middle of the picture?","criteria":["square","circle","triangle","star"]},
		  "red":{"type":"noul","instructions":"The shape in the middle is red."}}}`
	var fed [2]int
	for i := range fed {
		rec := ask(t, s, body)
		if rec.Code != 200 {
			t.Fatalf("status %d: %s", rec.Code, rec.Body)
		}
		var out struct {
			Answers map[string]struct {
				Choice        string             `json:"choice"`
				Probabilities map[string]float64 `json:"probabilities"`
				Noul          float64            `json:"noul"`
			} `json:"answers"`
			Usage struct {
				Input int `json:"input_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s", rec.Body)
		if out.Answers["shape"].Choice != "circle" || out.Answers["red"].Noul < 0.5 {
			t.Fatalf("%s", rec.Body)
		}
		fed[i] = out.Usage.Input
	}
	// The second request reads the questions again, and nothing of the state:
	// a system turn, a line and a picture, well over a hundred positions.
	if fed[0]-fed[1] < 100 {
		t.Fatalf("the second request fed %d positions and the first %d: the state was read again", fed[1], fed[0])
	}
}
