package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/engine"
	"github.com/ThiraSoft/golem/internal/heavy"
)

// screenshot is a picture of a screen's size that differs with seed, which is
// what a client sending a new capture every request sends.
func screenshot(seed int) string {
	img := image.NewRGBA(image.Rect(0, 0, 1280, 720))
	for y := 0; y < 720; y++ {
		for x := 0; x < 1280; x++ {
			img.Set(x, y, color.RGBA{uint8(x / 5), uint8(y / 3), uint8((x + y + seed*37) / 7), 255})
		}
	}
	for y := 300; y < 420; y++ {
		for x := 500 + seed*10; x < 640+seed*10; x++ {
			img.Set(x, y, color.RGBA{200, 30, 40, 255})
		}
	}
	var b bytes.Buffer
	png.Encode(&b, img)
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(b.Bytes())
}

// What a combat brain asks every few seconds: a persona and rules that never
// change, a history that grows slowly, a capture of the screen, the moment,
// and two questions of five and six options. The cost of an answer is taken
// three ways: from nothing, again with the same capture, and with a new
// capture behind the same rules and history, which is the request a client
// actually repeats.
func TestSystemOneCost(t *testing.T) {
	heavy.Skip(t, "a measurement, not a check")
	path, proj := os.Getenv("GOLEM_MODEL_12B"), os.Getenv("GOLEM_MMPROJ_12B")
	if path == "" || proj == "" {
		t.Skip("GOLEM_MODEL_12B and GOLEM_MMPROJ_12B are not both set")
	}
	m, err := engine.Open(path, 8192)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.OpenProjector(proj); err != nil {
		t.Fatal(err)
	}
	if err := m.UseVulkan(); err != nil {
		t.Fatal(err)
	}
	r := running(t, m.Forward)
	r.OnDevice()
	v, _ := m.Media()
	r.SetVision(v)
	if b, ok := m.Branches(); ok {
		r.SetBranches(b)
	}
	ctx := NewContext(r, m.Window, 8192, time.Now, 0)
	ctx.SetRing(m.WindowRing())
	s := NewServer(poolOf(NewGenerator(ctx, m.Vocab, m.Template, m.Vocabulary, 16)),
		m.Vocab, "12b", m.Template, m.Sampling)
	s.SetVision(v)

	var rules, history strings.Builder
	for i := range 30 {
		fmt.Fprintf(&rules, "Rule %d: when the boss casts ability %d, the party must react within %d seconds or take heavy damage; the tank keeps aggro and the healer stays in range.\n", i, i*7, 2+i%5)
	}
	for i := range 34 {
		fmt.Fprintf(&history, "[%02d:%02d] The party used order %s; the boss was at %d%% and the healer had %d%% mana.\n",
			i/6, (i*10)%60, []string{"HOLD", "ATTACK", "KEEP", "REGROUP"}[i%4], 100-i, 90-i)
	}
	request := func(seed int) string {
		state, _ := json.Marshal([]map[string]any{
			{"type": "text", "text": rules.String()},
			{"type": "text", "text": "History:\n" + history.String()},
			{"type": "image_url", "image_url": map[string]string{"url": screenshot(seed)}},
			{"type": "text", "text": "Now: the boss is at 12% health, the healer has 8% mana, two adds are up."},
		})
		return `{"system":"You are Nova, the combat brain of a party in an online game.","state":` + string(state) + `,
		"questions":{
		  "order":{"type":"choice","instructions":"Which order should the party follow now?","criteria":{
		    "HOLD":"keep the current positions","ATTACK":"all damage on the boss","KEEP":"keep the adds controlled",
		    "REGROUP":"gather behind the pillar","RETREAT":"leave the fight","FOCUS":"kill the adds first"}},
		  "gesture":{"type":"choice","instructions":"Which gesture should Nova make?","criteria":[
		    "attack the boss","attack the adds","heal the tank","heal","wait"]}}}`
	}
	timed := func(body string) (time.Duration, int) {
		start := time.Now()
		rec := ask(t, s, body)
		took := time.Since(start)
		if rec.Code != 200 {
			t.Fatalf("status %d: %s", rec.Code, rec.Body)
		}
		var out struct {
			Usage struct {
				Input int `json:"input_tokens"`
			} `json:"usage"`
		}
		json.Unmarshal(rec.Body.Bytes(), &out)
		return took, out.Usage.Input
	}
	median := func(seeds []int) (time.Duration, int) {
		var ds []time.Duration
		fed := 0
		for _, seed := range seeds {
			d, n := timed(request(seed))
			ds = append(ds, d)
			fed = n
		}
		slices.Sort(ds)
		return ds[len(ds)/2], fed
	}
	cold, coldFed := timed(request(0))
	same, sameFed := median([]int{0, 0, 0, 0, 0})
	fresh, freshFed := median([]int{1, 2, 3, 4, 5})
	t.Logf("from nothing: %s, %d positions fed", cold.Round(time.Millisecond), coldFed)
	t.Logf("same capture: %s, %d positions fed", same.Round(time.Millisecond), sameFed)
	t.Logf("new capture:  %s, %d positions fed", fresh.Round(time.Millisecond), freshFed)
}
