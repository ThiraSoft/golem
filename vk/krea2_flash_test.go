package vk

import (
	_ "embed"
	"encoding/binary"
	"testing"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_layout.comp -o shaders/krea2_layout.spv

//go:embed shaders/krea2_layout.spv
var krea2LayoutSPIRV []byte

// TestK2FlashLayout pins what shaders/krea2_flash.comp assumes of the
// driver: at a wave of thirty-two, a lane holds eight elements of a B or an
// accumulator sixteen-square, all in column lane%16, rows (lane/16)·8 on. A
// driver that lays them out otherwise fails here before the attention is
// wrong.
func TestK2FlashLayout(t *testing.T) {
	d := open(t)
	defer d.Close()
	if !d.Coopmat() {
		t.Skip("no cooperative matrices")
	}
	p, err := d.newPipeline(krea2LayoutSPIRV, 3, 0, 32, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	mh, err := d.Readback(512, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer mh.Close()
	mf, err := d.Readback(1024, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer mf.Close()
	o, err := d.Readback((16+3*1024)*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Close()
	for i := 0; i < 256; i++ {
		binary.LittleEndian.PutUint16(mh.Bytes()[2*i:], f32ToF16(float32(i)))
		mf.Floats()[i] = float32(i)
	}
	s, err := p.NewSet([]*Buffer{mh, mf, o})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := d.Submit(func(r *Recorder) { r.Dispatch(s, 1, nil) }); err != nil {
		t.Fatal(err)
	}
	out := o.Floats()
	if out[1] != 8 || out[2] != 8 || out[3] != 32 {
		t.Fatalf("B holds %v a lane, the accumulator %v, at a wave of %v; want 8, 8, 32", out[1], out[2], out[3])
	}
	for u, name := range []string{"B", "accumulator"} {
		for lane := 0; lane < 32; lane++ {
			for i := 0; i < 8; i++ {
				got := int(out[16+(u+1)*1024+lane*16+i])
				row, col := (lane/16)*8+i, lane%16
				if got != row*16+col {
					t.Fatalf("%s lane %d element %d is (%d, %d), want (%d, %d)", name, lane, i, got/16, got%16, row, col)
				}
			}
		}
	}
}
