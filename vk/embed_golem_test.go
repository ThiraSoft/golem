package vk

// SetEmbeddingGolem's shader reads a row a genuinely different way from the
// matvec kernels t4g_test.go already holds exact: instead of a dot product it
// runs the rotation forward once (its own inverse) and multiplies by the
// site's vector, producing one token's embedding rather than one output of a
// product. Nothing exercised that path before this test — the only assurance
// it had was that two end-to-end runs produced sane text, which is a
// measurement, not a test.
//
// This follows t4g_test.go's shape: dispatch the shader directly, without
// going through a Stack, and compare it exactly against nn.Matrix.Row, which
// takes the same table through the same rotation and vector on the CPU.

import (
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

func TestEmbedGolemMatchesCPU(t *testing.T) {
	for _, kind := range []nn.Quant{nn.T3G, nn.T4G, nn.T5G} {
		t.Run(kind.String(), func(t *testing.T) {
			testEmbedGolemMatchesCPU(t, kind)
		})
	}
}

func testEmbedGolemMatchesCPU(t *testing.T, kind nn.Quant) {
	const rows, cols = 4, 256
	d := open(t)
	defer d.Close()

	data, q := t4gMatrixAs(t, rows, cols, kind)
	pre := make([]float32, cols)
	for j := range pre {
		pre[j] = 1 / q[j]
	}

	m := nn.Matrix{Data: data, Quant: kind, Rows: rows, Cols: cols, Pre: pre, HadGroup: prepareD4GGroup}
	want := make([]float32, cols)

	var spirv []byte
	switch kind {
	case nn.T3G:
		spirv = embedT3GSPIRV
	case nn.T4G:
		spirv = embedT4GSPIRV
	case nn.T5G:
		spirv = embedT5GSPIRV
	}

	p, err := d.NewPipeline(spirv, 5, uint32(unsafe.Sizeof(embedPush{})))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()

	table, err := d.Upload(data)
	if err != nil {
		t.Fatal(err)
	}
	defer table.Close()
	steps, err := d.Upload(golemTable())
	if err != nil {
		t.Fatal(err)
	}
	defer steps.Close()
	preBuf, err := d.Upload(asBytes(pre))
	if err != nil {
		t.Fatal(err)
	}
	defer preBuf.Close()
	ids, err := d.Host(4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer ids.Close()
	out, err := d.Readback(cols*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	set, err := p.NewSet([]*Buffer{table, steps, ids, preBuf, out})
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()

	push := embedPush{cols: uint32(cols)}
	groups := uint32(cols / prepareD4GGroup)

	for r := 0; r < rows; r++ {
		m.Row(r, want)

		copy(ids.Bytes(), asBytesUint32([]uint32{uint32(int32(r))}))
		if err := set.Dispatch(groups, unsafe.Pointer(&push)); err != nil {
			t.Fatal(err)
		}
		got := out.Floats()[:cols]
		for j := 0; j < cols; j++ {
			if got[j] != want[j] {
				t.Fatalf("%s row %d, weight %d: the card reads %v, the processor %v",
					kind, r, j, got[j], want[j])
			}
		}
	}
}
