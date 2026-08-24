package nn

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// Every hand-encoded arm64 instruction, decoded from its bits and held against
// the comment beside it.
//
// The kernels emit SDOT, UDOT and FCVTL as WORD constants because Go's arm64
// assembler has no vector integer multiply and no vector half-to-single
// conversion. A mistyped constant is not a build error and often not a crash:
// it is a different valid instruction, and the signed and unsigned dot products
// are one bit apart. The comment is the only thing saying what was meant, and
// without this test nothing checks that it is true.
//
// The differential tests catch a wrong encoding too, but only on arm64 and only
// once someone runs an emulator. This one is plain text and arithmetic, so it
// runs everywhere — including on the x86 machine the port is written on.
func TestHandEncodedARM64InstructionsMatchTheirComments(t *testing.T) {
	files, err := filepath.Glob("*_arm64.s")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no arm64 assembly found; this test is watching nothing")
	}

	// WORD $0xdeadbeef  // SDOT V0.4S, V1.16B, V2.16B
	// The comment may be a /* */ one, because the macros need line continuations.
	line := regexp.MustCompile(`WORD\s+\$(0x[0-9a-f]{8})\s*/[/*]\s*(.*)$`)

	var checked int
	for _, file := range files {
		source, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		for i, text := range strings.Split(string(source), "\n") {
			m := line.FindStringSubmatch(text)
			if m == nil {
				if strings.Contains(text, "WORD $") {
					t.Errorf("%s:%d: hand-encoded word with no comment saying what it is", file, i+1)
				}
				continue
			}
			word, err := strconv.ParseUint(m[1][2:], 16, 32)
			if err != nil {
				t.Fatal(err)
			}
			claim := strings.TrimSpace(m[2])
			if at := strings.Index(claim, "*/"); at >= 0 {
				claim = strings.TrimSpace(claim[:at])
			}

			got := decodeARM64(uint32(word))
			if got == "" {
				t.Errorf("%s:%d: %s decodes to no instruction this test knows, but claims %q",
					file, i+1, m[1], claim)
				continue
			}
			if normalizeAsm(got) != normalizeAsm(claim) {
				t.Errorf("%s:%d: %s is %s, but the comment says %s", file, i+1, m[1], got, claim)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Error("no hand-encoded instructions found; either they are gone or the pattern stopped matching")
	}
	t.Logf("%d hand-encoded instructions decoded and matched", checked)
}

// decodeARM64 returns the instruction a word encodes, for the few shapes these
// kernels use, or "" for anything else. The field layouts are the architecture
// manual's.
func decodeARM64(w uint32) string {
	bits := func(hi, lo uint) uint32 { return (w >> lo) & (1<<(hi-lo+1) - 1) }

	// SDOT/UDOT (vector): 0 Q U 01110 size 0 Rm 1001 01 Rn Rd
	if bits(31, 31) == 0 && bits(28, 24) == 0b01110 && bits(21, 21) == 0 &&
		bits(15, 12) == 0b1001 && bits(11, 10) == 0b01 {
		if bits(30, 30) == 1 && bits(23, 22) == 0b10 {
			name := "SDOT"
			if bits(29, 29) == 1 {
				name = "UDOT"
			}
			return name + " V" + itoa(bits(4, 0)) + ".4S, V" + itoa(bits(9, 5)) +
				".16B, V" + itoa(bits(20, 16)) + ".16B"
		}
	}

	// FCVTL and FCVTN (vector): 0 Q 0 01110 0 sz 10000 1011x 10 Rn Rd, where the
	// x is 1 for the widening and 0 for the narrowing.
	if bits(31, 31) == 0 && bits(29, 29) == 0 && bits(28, 24) == 0b01110 &&
		bits(23, 23) == 0 && bits(21, 17) == 0b10000 && bits(16, 13) == 0b1011 &&
		bits(11, 10) == 0b10 && bits(22, 22) == 0 {
		two, narrow := "", "4H"
		if bits(30, 30) == 1 {
			two, narrow = "2", "8H"
		}
		if bits(12, 12) == 1 {
			return "FCVTL" + two + " V" + itoa(bits(4, 0)) + ".4S, V" + itoa(bits(9, 5)) + "." + narrow
		}
		return "FCVTN" + two + " V" + itoa(bits(4, 0)) + "." + narrow + ", V" + itoa(bits(9, 5)) + ".4S"
	}

	return ""
}

func itoa(v uint32) string { return strconv.FormatUint(uint64(v), 10) }

// normalizeAsm drops the spacing and punctuation the comments are inconsistent
// about, so the comparison is about registers and mnemonics.
func normalizeAsm(s string) string {
	var b strings.Builder
	for _, r := range strings.ToUpper(s) {
		if r != ' ' && r != ',' && r != '\t' {
			b.WriteRune(r)
		}
	}
	return b.String()
}
