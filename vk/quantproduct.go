package vk

// One weight format, one pipeline, and the same four buffers whichever it is.
//
// Every projection on this card is the same product: a quantized matrix against
// the Q8_0 form of whatever feeds it, answering a float. What differs between
// Q4_0, Q4_K and Q6_K is thirty lines of unpacking inside two shaders — the
// mat-vec a token takes and the tiled product a prompt takes — and nothing
// outside them. So a caller that has a matrix in one of these formats asks for
// its pipeline and binds the same set it always did.
//
// That is the whole of what made a K-quant unreadable here. The kernels were
// not missing so much as the *choice* was: vk/attention.go uploaded four
// matrices through splitQ4_0 with no idea what they were, vk/mixture.go did the
// same for three more, and both checked a length rather than a type. Eighteen
// bytes to a block of thirty-two is Q4_0 and it is also Q4_K, so half of those
// checks would have passed on the wrong format and answered with noise.

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// The mat-vec over a Q4_K matrix and a Q8_0 activation, at the widths a token
// and a short pass take. shaders/matvec.comp under -DQ4K.
//
//go:generate glslc -O -DQ4K --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q4kq8.spv
//go:generate glslc -O -DQ4K -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q4kq8_2.spv
//go:generate glslc -O -DQ4K -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q4kq8_4.spv
//go:generate glslc -O -DQ4K -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q4kq8_8.spv
//go:generate glslc -O -DQ4K -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q4kq8_16.spv
//go:generate glslc -O -DQ6K --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q6kq8.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q6kq8_2.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q6kq8_4.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q6kq8_8.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q6kq8_16.spv
//go:generate glslc -O -DQ4K -DCOLUMNS=32 -DBN=32 -DBM=64 -DBK=128 -DWAVE_M=4 -DWAVE_N=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q4k32.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=32 -DBN=32 -DBM=64 -DBK=128 -DWAVE_M=4 -DWAVE_N=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q6k32.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=64 -DBN=64 -DBM=128 -DBK=128 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q6k64.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=128 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q6k128.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=256 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q6k256.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=512 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q6k512.spv

//go:embed shaders/matvec_q4kq8.spv
var matvecQ4KQ8SPIRV []byte

//go:embed shaders/matvec_q4kq8_2.spv
var matvecQ4KQ8_2SPIRV []byte

//go:embed shaders/matvec_q4kq8_4.spv
var matvecQ4KQ8_4SPIRV []byte

//go:embed shaders/matvec_q4kq8_8.spv
var matvecQ4KQ8_8SPIRV []byte

//go:embed shaders/matvec_q4kq8_16.spv
var matvecQ4KQ8_16SPIRV []byte

//go:embed shaders/matvec_q6kq8.spv
var matvecQ6KQ8SPIRV []byte

//go:embed shaders/matvec_q6kq8_2.spv
var matvecQ6KQ8_2SPIRV []byte

//go:embed shaders/matvec_q6kq8_4.spv
var matvecQ6KQ8_4SPIRV []byte

//go:embed shaders/matvec_q6kq8_8.spv
var matvecQ6KQ8_8SPIRV []byte

//go:embed shaders/matvec_q6kq8_16.spv
var matvecQ6KQ8_16SPIRV []byte

//go:embed shaders/matmul_coop_q4k32.spv
var matmulCoopQ4K32SPIRV []byte

//go:embed shaders/matmul_coop_q6k32.spv
var matmulCoopQ6K32SPIRV []byte

// QuantReadable says whether a projection stored this way has a kernel here.
//
// It is asked before a matrix is uploaded rather than inferred from its length,
// because the lengths collide: eighteen bytes to a block of thirty-two is Q4_0
// and Q4_K both, so a check that passed would have decoded one as the other and
// answered fluently. Two faults of exactly that shape were found in this
// repository, and both were only caught because they happened to overrun the
// tensor instead.
func QuantReadable(q nn.Quant) bool {
	switch q {
	case nn.Q4_0, nn.Q4_K, nn.Q6_K:
		return true
	}
	return false
}

// quantRowBytes is what one row of that many inputs occupies once the packing
// for its format has had it.
func quantRowBytes(q nn.Quant, cols int) (int, error) {
	switch q {
	case nn.Q4_0:
		return rowBytesQ4_0(cols), nil
	case nn.Q4_K:
		return rowBytesQ4_K(cols), nil
	case nn.Q6_K:
		return rowBytesQ6_K(cols), nil
	}
	return 0, fmt.Errorf("vk: there is no projection kernel for %s", q)
}

// quantLayout is a matrix in the layout its kernels read, checked against the
// length the format says it should have.
func quantLayout(q nn.Quant, data []byte, rows, cols int) ([]byte, error) {
	if q != nn.Q4_0 && cols%nn.SuperBlock != 0 {
		return nil, fmt.Errorf("vk: a %s row needs a multiple of %d columns, given %d", q, nn.SuperBlock, cols)
	}
	if cols%nn.QuantBlock != 0 {
		return nil, fmt.Errorf("vk: a quantized row needs a multiple of %d columns, given %d", nn.QuantBlock, cols)
	}
	// The file's own count, not the packing's: Q6_K's rows are rounded up to a
	// word on the way to the card and the file does not round them.
	fileRow := 0
	switch q {
	case nn.Q4_0:
		fileRow = rowBytesQ4_0(cols)
	case nn.Q4_K:
		fileRow = rowBytesQ4_K(cols)
	case nn.Q6_K:
		fileRow = cols / nn.SuperBlock * 210
	default:
		return nil, fmt.Errorf("vk: there is no projection kernel for %s", q)
	}
	if want := rows * fileRow; len(data) != want {
		return nil, fmt.Errorf("vk: %d rows of %d columns in %s need %d bytes, given %d", rows, cols, q, want, len(data))
	}
	switch q {
	case nn.Q4_0:
		return splitQ4_0(data, rows, cols), nil
	case nn.Q4_K:
		return splitQ4_K(data, rows, cols), nil
	default:
		return splitQ6_K(data, rows, cols), nil
	}
}

// newQuantProduct builds the pipeline that reads one weight format: the mat-vec
// at the narrow widths, the tiled product at the wide ones.
//
// The narrow widths differ between the formats and that is deliberate. Q4_0 is
// built at one and eight here because that is what it has always been built at
// and the tiled product takes everything past it; the K-quants are built at
// one, two, four, eight and sixteen because on a card with no matrix cores they
// have no tiled form and the mat-vec is the whole of the answer.
func newQuantProduct(d *Device, q nn.Quant, coop bool) (*Pipeline, error) {
	var base []byte
	var narrow []struct {
		columns int
		spirv   []byte
	}
	var tiled []struct {
		columns int
		spirv   []byte
	}
	switch q {
	case nn.Q4_0:
		base = matvecSPIRV
		narrow = append(narrow, struct {
			columns int
			spirv   []byte
		}{smallColumns, matvecWideSPIRV})
	case nn.Q4_K:
		base = matvecQ4KQ8SPIRV
		for _, w := range []struct {
			columns int
			spirv   []byte
		}{{2, matvecQ4KQ8_2SPIRV}, {4, matvecQ4KQ8_4SPIRV}, {smallColumns, matvecQ4KQ8_8SPIRV}, {16, matvecQ4KQ8_16SPIRV}} {
			narrow = append(narrow, w)
		}
	case nn.Q6_K:
		base = matvecQ6KQ8SPIRV
		for _, w := range []struct {
			columns int
			spirv   []byte
		}{{2, matvecQ6KQ8_2SPIRV}, {4, matvecQ6KQ8_4SPIRV}, {smallColumns, matvecQ6KQ8_8SPIRV}, {16, matvecQ6KQ8_16SPIRV}} {
			narrow = append(narrow, w)
		}
	default:
		return nil, fmt.Errorf("vk: there is no projection kernel for %s", q)
	}

	if coop {
		switch q {
		case nn.Q4_0:
			tiled = []struct {
				columns int
				spirv   []byte
			}{
				{tiledColumns, matmulCoop32SPIRV}, {64, matmulCoop64SPIRV},
				{128, matmulCoop128SPIRV}, {256, matmulCoop256SPIRV}, {wideColumns, matmulCoop512SPIRV},
			}
		case nn.Q4_K:
			tiled = []struct {
				columns int
				spirv   []byte
			}{
				{tiledColumns, matmulCoopQ4K32SPIRV}, {64, matmulCoopQ4K64SPIRV},
				{128, matmulCoopQ4K128SPIRV}, {256, matmulCoopQ4K256SPIRV}, {wideColumns, matmulCoopQ4K512SPIRV},
			}
		case nn.Q6_K:
			tiled = []struct {
				columns int
				spirv   []byte
			}{
				{tiledColumns, matmulCoopQ6K32SPIRV}, {64, matmulCoopQ6K64SPIRV},
				{128, matmulCoopQ6K128SPIRV}, {256, matmulCoopQ6K256SPIRV}, {wideColumns, matmulCoopQ6K512SPIRV},
			}
		}
	} else if q == nn.Q4_0 {
		// The integer tiles, which only Q4_0 has. A K-quant on a card without
		// matrix cores runs its mat-vec at sixteen columns a dispatch, which is
		// slower and right; the alternative was refusing the model.
		tiled = []struct {
			columns int
			spirv   []byte
		}{
			{tiledColumns, matmulWide32SPIRV}, {64, matmulWide64SPIRV},
			{128, matmulWidest128SPIRV}, {256, matmulWidest256SPIRV}, {wideColumns, matmulWide()},
		}
	}

	p, err := d.newPipeline(base, 4, uint32(unsafe.Sizeof(moePush{})), 0, nil)
	if err != nil {
		return nil, err
	}
	for _, w := range narrow {
		if err := p.Wide(w.columns, w.spirv); err != nil {
			p.Close()
			return nil, err
		}
	}
	wave := uint32(0)
	if coop {
		wave = coopmatWave
	}
	for _, w := range tiled {
		if err := p.WideWave(w.columns, w.spirv, wave); err != nil {
			p.Close()
			return nil, err
		}
	}
	return p, nil
}

// quantProducts is a lazily built pipeline per format. A model uses one or two
// of them and there is no reason to compile the rest: each is ten binaries.
type quantProducts struct {
	d     *Device
	coop  bool
	byQ   map[nn.Quant]*Pipeline
	order []*Pipeline
}

func newQuantProducts(d *Device, coop bool) *quantProducts {
	return &quantProducts{d: d, coop: coop, byQ: map[nn.Quant]*Pipeline{}}
}

func (s *quantProducts) get(q nn.Quant) (*Pipeline, error) {
	if p, ok := s.byQ[q]; ok {
		return p, nil
	}
	p, err := newQuantProduct(s.d, q, s.coop)
	if err != nil {
		return nil, err
	}
	s.byQ[q] = p
	s.order = append(s.order, p)
	return p, nil
}

func (s *quantProducts) Close() {
	for _, p := range s.order {
		p.Close()
	}
	s.order, s.byQ = nil, map[nn.Quant]*Pipeline{}
}
