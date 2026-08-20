package wav

import (
	"bytes"
	"math"
	"testing"
)

// What Write produces, Read must give back. It is the only round trip the
// package owes anyone.
func TestRoundTrip(t *testing.T) {
	want := make([]float32, 480)
	for i := range want {
		want[i] = float32(math.Sin(float64(i) * 0.05))
	}

	var buf bytes.Buffer
	if err := Write(&buf, want, 24000); err != nil {
		t.Fatal(err)
	}
	got, rate, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 24000 {
		t.Errorf("rate = %d, want 24000", rate)
	}
	if len(got) != len(want) {
		t.Fatalf("%d samples, want %d", len(got), len(want))
	}
	// Sixteen-bit quantization is the whole of the loss, and it is worth two
	// steps rather than one: Write scales by 32767 and truncates toward zero,
	// Read divides by 32768, which is the range an int16 actually spans. Half a
	// step for the truncation, and up to one more for the two scales not being
	// the same number.
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > 2.0/32768 {
			t.Fatalf("sample %d: %v, want %v (gap %v)", i, got[i], want[i], d)
		}
	}
}

func TestReadRejectsWhatIsNotAWav(t *testing.T) {
	if _, _, err := Read(bytes.NewReader([]byte("this is not a wav file at all"))); err == nil {
		t.Fatal("expected an error")
	}
}

// A chunk the file did not need — the "LIST" some editors write — must be
// stepped over, not read as samples.
func TestReadSkipsUnknownChunks(t *testing.T) {
	var buf bytes.Buffer
	if err := Write(&buf, []float32{0.5, -0.5}, 24000); err != nil {
		t.Fatal(err)
	}
	raw := buf.Bytes()
	// Splice a LIST chunk between the format and the data chunks. The header
	// Write produces is fixed: 12 bytes of RIFF, then 24 of "fmt ".
	const cut = 36
	extra := append([]byte("LIST"), 4, 0, 0, 0, 'I', 'N', 'F', 'O')
	spliced := append(append(append([]byte{}, raw[:cut]...), extra...), raw[cut:]...)

	got, _, err := Read(bytes.NewReader(spliced))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("%d samples, want 2", len(got))
	}
}
