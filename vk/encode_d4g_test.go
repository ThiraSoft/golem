package vk

import (
	"math/rand"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/compress"
	"github.com/ThiraSoft/golem/nn"
)

// The card and the processor must write the same file. The step search is a
// scan over codes with an early abandon and the shell fallback is a scan over
// pulls, so both sides have to make the same choice at every tie, not merely a
// choice of the same quality — a file is bytes.
func TestEncodeD4GMatchesCPU(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const rows, cols, group = 96, 512, 128
	r := rand.New(rand.NewSource(9))
	w := make([]float32, rows*cols)
	for i := range w {
		v := r.NormFloat64() * 0.02
		// A row of pure noise never leaves the shell. Real weights do, and the
		// fallback is where the two implementations are most likely to part.
		if r.Intn(64) == 0 {
			v *= 12
		}
		w[i] = float32(v)
	}
	q := make([]float32, cols)
	for j := range q {
		s := float32(0.4 + r.Float64())
		if r.Intn(2) == 0 {
			s = -s
		}
		q[j] = s
	}

	p := compress.D4Params{Beta: 2, ScaleBlock: nn.D4SubBlock, HadGroup: group,
		Bits: nn.D4Bits, SearchScale: true}
	want := compress.EncodeD4G(w, rows, cols, q, p, nil)

	e, err := NewD4GEncoder(d, nn.D4Bits, rows*cols)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	gotCodes := make([]uint16, rows*cols/4)
	gotSteps := make([]byte, rows*cols/nn.D4SubBlock)
	if err := e.Encode(w, rows, cols, q, D4GEncodeParams{
		HadGroup: group, Beta: 2, SpanLo: 0.45, SpanHi: 2.4,
	}, gotCodes, gotSteps); err != nil {
		t.Fatal(err)
	}

	got := compress.PackD4G(gotCodes, gotSteps, rows, cols, nn.D4Bits)
	bad := 0
	for i := range want {
		if want[i] != got[i] {
			if bad == 0 {
				t.Errorf("byte %d of %d: %#02x on the processor, %#02x on the card", i, len(want), want[i], got[i])
			}
			bad++
		}
	}
	if bad != 0 {
		t.Fatalf("%d of %d bytes differ", bad, len(want))
	}
}

// A matrix that is neither scaled nor rotated goes through the same kernel with
// nothing in front of it, and that path has to agree too.
func TestEncodeD4GUnrotatedMatchesCPU(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const rows, cols = 32, 256
	r := rand.New(rand.NewSource(3))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(r.NormFloat64() * 0.05)
	}
	p := compress.D4Params{Beta: 2, ScaleBlock: nn.D4SubBlock, Bits: nn.D4Bits, SearchScale: true}
	want := compress.EncodeD4G(w, rows, cols, nil, p, nil)

	e, err := NewD4GEncoder(d, nn.D4Bits, rows*cols)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	gotCodes := make([]uint16, rows*cols/4)
	gotSteps := make([]byte, rows*cols/nn.D4SubBlock)
	if err := e.Encode(w, rows, cols, nil, D4GEncodeParams{Beta: 2, SpanLo: 0.45, SpanHi: 2.4}, gotCodes, gotSteps); err != nil {
		t.Fatal(err)
	}
	got := compress.PackD4G(gotCodes, gotSteps, rows, cols, nn.D4Bits)
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("byte %d of %d: %#02x against %#02x", i, len(want), want[i], got[i])
		}
	}
}

// What the whole point of this was.
func TestEncodeD4GIsFasterThanTheProcessor(t *testing.T) {
	if testing.Short() {
		t.Skip("timing")
	}
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const rows, cols, group = 2048, 5120, 128
	r := rand.New(rand.NewSource(11))
	w := make([]float32, rows*cols)
	for i := range w {
		v := r.NormFloat64() * 0.02
		if r.Intn(64) == 0 {
			v *= 12
		}
		w[i] = float32(v)
	}
	q := make([]float32, cols)
	for j := range q {
		q[j] = 1
		if r.Intn(2) == 0 {
			q[j] = -1
		}
	}
	p := compress.D4Params{Beta: 2, ScaleBlock: nn.D4SubBlock, HadGroup: group,
		Bits: nn.D4Bits, SearchScale: true}

	t0 := time.Now()
	compress.EncodeD4G(w, rows, cols, q, p, nil)
	cpu := time.Since(t0)

	e, err := NewD4GEncoder(d, nn.D4Bits, rows*cols)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	codes := make([]uint16, rows*cols/4)
	steps := make([]byte, rows*cols/nn.D4SubBlock)
	// Once to warm the clocks, once to time.
	if err := e.Encode(w, rows, cols, q, D4GEncodeParams{HadGroup: group, Beta: 2, SpanLo: 0.45, SpanHi: 2.4}, codes, steps); err != nil {
		t.Fatal(err)
	}
	t1 := time.Now()
	if err := e.Encode(w, rows, cols, q, D4GEncodeParams{HadGroup: group, Beta: 2, SpanLo: 0.45, SpanHi: 2.4}, codes, steps); err != nil {
		t.Fatal(err)
	}
	gpu := time.Since(t1)

	n := float64(rows * cols)
	t.Logf("%.1f M weights: processor %s (%.2f M/s), card %s (%.2f M/s), %.1fx",
		n/1e6, cpu.Round(time.Millisecond), n/cpu.Seconds()/1e6,
		gpu.Round(time.Millisecond), n/gpu.Seconds()/1e6, float64(cpu)/float64(gpu))
	t.Logf("26.0 G weights: processor %s, card %s",
		time.Duration(float64(cpu)*26e9/n).Round(time.Second),
		time.Duration(float64(gpu)*26e9/n).Round(time.Second))
	if gpu >= cpu {
		t.Errorf("the card is not faster: %s against %s", gpu, cpu)
	}
}
