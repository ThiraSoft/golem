package vk

// The feed-forward half of a mixture block, on the card.
//
// This is where the bytes are. A token of the 26B A4B reads about 2.3
// gigabytes; the logit head is 0.6 of them, the experts 0.8, and the shared
// branch beside them 0.3. The experts are read eight matrices at a time out of
// a hundred and twenty-eight, thirty times over, and on the CPU it is all
// bandwidth — the same 37 gigabytes a second the head was stuck at.
//
// Both branches are here because they are one submission. They leave the same
// residual under two different norms and their outputs are added, so nothing
// in either waits on the other: the card can run them together, and a
// submission costs sixty-three microseconds whatever is in it. Two of them a
// block would be four milliseconds a token spent asking rather than computing.
//
// The whole stack is resident — 11.96 gibibytes of experts on this checkpoint
// and 0.3 more of shared branches, which is why a card with sixteen is the
// smallest one that can do this. The routing stays on the CPU: it reads the
// residual, which the CPU already has, and choosing eight of a hundred and
// twenty-eight is a hundred and twenty-eight comparisons.

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"os"
	"strconv"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_gateup.comp -o shaders/moe_gateup.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_act.comp -o shaders/moe_act.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_down.comp -o shaders/moe_down.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec.spv

// The prompt path of the expert branch, which reads the stack by expert
// rather than by column. shaders/moe_scatter.comp says why.
//
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_scatter.comp -o shaders/moe_scatter.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_id_combine.comp -o shaders/moe_id_combine.spv

// The down projection of that path, which is the tiled product itself read by
// expert. shaders/matmul.comp's BYID section says what it replaced.
//
//go:generate glslc -O -DCOLUMNS=16 -DBYID -DKSTEP=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul_id16.spv

//go:embed shaders/matmul_id16.spv
var matmulByIDSPIRV []byte

//go:embed shaders/matmul_coop_id.spv
var matmulCoopByIDSPIRV []byte

//go:embed shaders/moe_scatter.spv
var moeScatterSPIRV []byte

//go:embed shaders/moe_id_combine.spv
var moeIDCombineSPIRV []byte

//go:embed shaders/moe_gateup.spv
var moeGateUpSPIRV []byte

// The activation and the quantization on their own, for the pass that takes
// its gate and up products from the tiled kernel instead. See
// shaders/moe_act.comp.
//
//go:generate glslc -O -DBYID --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_act.comp -o shaders/moe_act_id.spv
//go:embed shaders/moe_act.spv
var moeActSPIRV []byte

//go:embed shaders/moe_act_id.spv
var moeActByIDSPIRV []byte

//go:embed shaders/moe_down.spv
var moeDownSPIRV []byte

// matvecSPIRV is the plain Q4_0 product, which the shared branch's down
// projection and the attention's four projections both want.
//
//go:embed shaders/matvec.spv
var matvecSPIRV []byte

// The same two kernels built for a batch of columns. shaders/matvec.comp says
// why the count is compiled in rather than pushed.
//
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec8.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_gateup.comp -o shaders/moe_gateup8.spv
//go:generate glslc -O -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_gateup.comp -o shaders/moe_gateup16.spv
//go:generate glslc -O -DCOLUMNS=32 --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_gateup.comp -o shaders/moe_gateup32.spv

//go:embed shaders/matvec8.spv
var matvecWideSPIRV []byte

//go:embed shaders/moe_gateup8.spv
var moeGateUpWideSPIRV []byte

//go:embed shaders/moe_gateup16.spv
var moeGateUpMidSPIRV []byte

//go:embed shaders/moe_gateup32.spv
var moeGateUpWidestSPIRV []byte

// gateColumns is how many columns one dispatch of the gate kernel answers.
//
// It is not the width of the pass, and the difference is measured. The kernel
// keeps one accumulator a column in registers across its whole walk of the
// shared dimension, so its width is bounded by what a wave can hold rather
// than by shared memory: at eight columns a pass of thirty-two runs it four
// times and reads the gate and up matrices four times, and that is still
// faster than reading them once at thirty-two. See shaders/moe_gateup.comp.
const gateColumns = 8

// The tiled product, which is the shape a prompt wants where the mat-vec is
// the shape a token wants. shaders/matmul.comp says why.
//
//go:embed shaders/matmul32.spv
var matmulWide32SPIRV []byte

//go:embed shaders/matmul64.spv
var matmulWide64SPIRV []byte

//go:embed shaders/matmul128.spv
var matmulWidest128SPIRV []byte

//go:embed shaders/matmul256.spv
var matmulWidest256SPIRV []byte

//go:embed shaders/matmul512.spv
var matmulWidest512SPIRV []byte

//go:embed shaders/matmul_reduce.spv
var matmulReduceSPIRV []byte

//go:embed shaders/matmul8.spv
var matmulSmallSPIRV []byte

// The same product on the matrix cores, where the device has them.
//
//go:embed shaders/matmul_coop32.spv
var matmulCoop32SPIRV []byte

//go:embed shaders/matmul_coop64.spv
var matmulCoop64SPIRV []byte

//go:embed shaders/matmul_coop128.spv
var matmulCoop128SPIRV []byte

//go:embed shaders/matmul_coop256.spv
var matmulCoop256SPIRV []byte

//go:embed shaders/matmul_coop512.spv
var matmulCoop512SPIRV []byte

// coopTile is the cooperative matrix's own size, and the granularity a
// workgroup of shaders/matmul_coop.comp can be trusted with: a matrix whose
// row count is not a multiple of it cannot use the kernel at all.
const coopTile = 16

// expertsUsed is what the mixture's down kernel is written for: its workgroup
// is eight outputs by eight experts. A checkpoint that chose a different
// number would need the shape changed, not a constant.
const expertsUsed = 8

// downOuts is how many outputs shaders/moe_down.comp writes per workgroup.
const downOuts = 8

// matvecOuts is shaders/matvec.comp's, which serves the shared branch's down
// projection and every projection of an attention.
const matvecOuts = 16

// A Mixture is every feed-forward matrix of a model, resident, plus the
// kernels that read them and the small buffers a token passes through.
type Mixture struct {
	tl *Timeline // set by Profile, nil everywhere else
	d  *Device

	dim, ffn, dense, experts int
	act                      Activation
	coop                     bool
	// byExpertFrom is the width at which Record stops answering a column at a
	// time. Read once here rather than in Record, so that a test setting the
	// environment before it opens a model gets the path it asked for and the
	// recordings are not built against two answers.
	byExpertFrom int

	gateUp, down, denseDown *Pipeline
	// The prompt path of the expert branch, which reads the stack by expert
	// rather than by column: idProduct serves both gate/up and down projections,
	// idActivate runs the activation over the first half's output, scatter
	// builds the lists they walk, and idCombine adds a column's eight slots
	// back together. shaders/moe_scatter.comp says why a prompt wants this.
	// idProduct is shaders/matmul.comp (or matmul_coop.comp) built with BYID.
	idProduct          *Pipeline
	scatter, idCombine *Pipeline
	scatterSet         *Set
	combineSet         *Set
	counts, pairs      *Buffer
	// plan is the grid the by-expert product is dispatched against: one entry
	// per expert and stretch of idProductBN that has work, written by the
	// scatter beside the counts because only the card knows how many there
	// are. See shaders/moe_scatter.comp.
	plan  *Buffer
	dpart *Buffer
	// reduce folds the slices of a split shared-branch down projection, and
	// splitDown is how many there are. matmulSplit says which matrices want
	// it: this one has the stack's fewest rows and the most weight behind
	// them.
	reduce    *Pipeline
	splitDown int
	reduceSet *Set
	gelu      *Buffer // ggml's GELU table, uploaded once
	zero      *Buffer // one identifier, always zero, for the shared branch

	// The shared branch's gate and up as a tiled product rather than a fused
	// mat-vec: gateOut is where both halves land, activate is the kernel that
	// reads them back, and actSet is its one set for the whole mixture — the
	// weights are the only thing in that stage that belongs to a block.
	// Record says at what width this path takes over.
	activate   *Pipeline
	idActivate *Pipeline
	gateOut    *Buffer
	actSet     *Set
	idActSet   *Set

	// The expert branch's traffic, reused by every block.
	xq, xs, ids, cw, out *Buffer
	aq, as               *Buffer // the intermediate, which never leaves the card

	// The shared branch's, which is the same shape with one expert.
	dxq, dxs, dout *Buffer
	// dxf is the shared branch's input as floats, which only a Golem branch
	// reads, and which is nil until one is added.
	dxf       *Buffer
	doutParts *Buffer // the slices of a split down projection
	daq, das  *Buffer

	blocks []*mixtureBlock
}

// A mixtureBlock is one block's matrices and the bindings that read them.
type mixtureBlock struct {
	// golem is the shared branch when it is in that format, and nil when it is
	// Q4_0. vk/mixture_golem.go is all of it.
	golem *GolemFFN

	gateUp, down           *Buffer
	denseGateUp, denseDown *Buffer
	setGateUp, setDown     *Set
	setDenseUp, setDenseDn *Set
	setDenseProd           *Set // the gate and up through the tiled product, for a wide pass
	setDenseDnParts        *Set // the same, writing the slices of a split product
	setIDProd, setIDDown   *Set // the expert branch read by expert, for a prompt
}

// moePush is what all three kernels take. matvec.comp reads the first three
// fields and ignores the activation, which is spent before it runs.
type moePush struct {
	dim  uint32
	ffn  uint32
	used uint32
	act  uint32
	// col is the first column of the pass a dispatch answers, which only the
	// gate kernel reads: it is the one kernel that cannot be built at the full
	// width of a pass, so a wide pass runs it more than once at an offset.
	// Every other kernel declares four uints and ignores this one.
	col uint32
	// split is how many slices of the shared dimension a tiled product is cut
	// into, and only shaders/matmul.comp reads it. One means the whole of it in
	// one workgroup and the answer written where the caller wants it; more
	// means each slice writes its own copy and shaders/matmul_reduce.comp adds
	// them. Every other kernel declares fewer uints and ignores this one.
	split uint32
	// cap is the room in one expert's list of columns, which the two kernels
	// that read those lists index by. Every other kernel declares fewer uints
	// and ignores it.
	cap uint32
	// bycol is 1 when the input activation is indexed by column (the gate/up
	// projection) and 0 when indexed by pair (the down projection).
	bycol uint32
}

// idProductBN is the BN shaders/matmul.comp is built at for the by-expert
// down projection, and the stretch of one expert's list a workgroup answers.
// It is not the pass's width: a hundred and twenty-eight experts share eight
// choices from each column, so an expert's list holds columns*used/experts
// entries — sixteen at a pass of two hundred and fifty-six.
const (
	idProductBN     = 16
	idProductCoopBN = 64
	idProductCoopBM = 128
)

// idPlanMax is how many entries that plan can hold for a pass of that width,
// which is what the dispatch has to be sized for: one entry per bn pairs, plus
// one an expert for the stretch that does not fill.
//
// bn is the caller's, not a constant, because the two products have different
// ones: shaders/matmul.comp's BYID build answers idProductBN entries and
// shaders/matmul_coop.comp's answers idProductCoopBN. Sized with the smaller
// where the larger is dispatched, the grid is twice what the plan can hold and
// half the workgroups read one uint and return — which is safe, and was
// measurably there. Sized the other way round it would drop the tail of every
// expert's list, which is not.
func idPlanMax(experts, columns, bn int) int {
	return experts + columns*expertsUsed/bn
}

// byExpertFrom is the width at which a mixture stops answering a column at a
// time and starts reading the whole expert stack once for the pass.
//
// The two costs are eight matrices a column against every expert the model
// has, so they cross where the pass is about a sixteenth of the stack wide.
// Measured on a 26B A4B and an RX 9070 XT, positions a second by width, each
// path forced:
//
//	width        2    4    8   12   16   24    32    64
//	by column  184  328  557  778  990 1240  1370  1516
//	by expert  106  207  441  720  943 1228  1734  2765
//
// They cross between twenty-four and thirty-two. Below it the by-expert path
// reads a hundred and twenty-eight experts to answer a handful of pairs, which
// is what a server drawing a token for each of two conversations was paying.
//
// GOLEM_MOE_BY_EXPERT_FROM moves it, which is how the tests reach the other
// path on a six-token fixture: neither is the fallback and both have to be
// right, so both are run against the same recording.
const byExpertFrom = 32

// byExpertWidth is that, with the environment's override if there is one.
func byExpertWidth() int {
	v := os.Getenv("GOLEM_MOE_BY_EXPERT_FROM")
	if v == "" {
		return byExpertFrom
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		return byExpertFrom
	}
	return n
}

// idBN is the entries of a list one workgroup of the by-expert product
// answers, which is the kernel's own BN and has to reach the scatter, the plan
// and the dispatch as one number.
func idBN(coop bool) int {
	if coop {
		return idProductCoopBN
	}
	return idProductBN
}

// scatterPush is shaders/moe_scatter.comp's, which counts rather than
// multiplies and takes none of the shapes the others do.
// It is padded to moePush's size because that is what every pipeline here
// declares its push range as, and a recorder copies the range and not the
// struct.
type scatterPush struct {
	columns uint32
	used    uint32
	experts uint32
	cap     uint32
	bn      uint32
	_       [3]uint32
}

// combinePairsPush is shaders/moe_id_combine.comp's.
type combinePairsPush struct {
	dim     uint32
	used    uint32
	columns uint32
	_       [5]uint32
}

// The padding above is not decoration, and these say so at compile time: a
// pipeline here declares moePush's size as its push range and a recorder
// copies the range, not the struct, so a shorter struct is read past its end.
// Either subtraction goes negative — and an untyped constant may not — the
// moment the three sizes stop agreeing.
const (
	_ = unsafe.Sizeof(moePush{}) - unsafe.Sizeof(scatterPush{})
	_ = unsafe.Sizeof(scatterPush{}) - unsafe.Sizeof(moePush{})
	_ = unsafe.Sizeof(moePush{}) - unsafe.Sizeof(combinePairsPush{})
	_ = unsafe.Sizeof(combinePairsPush{}) - unsafe.Sizeof(moePush{})
)

// An Activation is what a gated feed forward puts on its gate. Gemma 4 looks
// ggml's GELU up in a table; Qwen3 evaluates a SiLU. There is no third.
type Activation uint32

const (
	GELU Activation = iota
	SiLU
)

// NewMixture builds the kernels and the shared buffers. AddBlock then uploads
// one block at a time, so that a caller can report progress over twelve
// gibibytes rather than disappear into them.
//
// ffn is one expert's width and dense the shared branch's, which are not the
// same number: 704 against 2112 on this checkpoint.
func NewMixture(d *Device, dim, ffn, dense, experts, used int, act Activation) (*Mixture, error) {
	if experts > 0 && used != expertsUsed {
		return nil, fmt.Errorf("vk: the expert kernels are written for %d experts a token, this model uses %d", expertsUsed, used)
	}
	shapes := []int{dim, dense}
	if experts > 0 {
		shapes = append(shapes, ffn)
	}
	for _, n := range shapes {
		if n%nn.QuantBlock != 0 {
			return nil, fmt.Errorf("vk: feed-forward shapes must be multiples of %d, given %d", nn.QuantBlock, n)
		}
	}
	if dim%downOuts != 0 {
		return nil, fmt.Errorf("vk: the down kernel writes %d outputs at a time, and %d is not a multiple of it", downOuts, dim)
	}
	coop := d.Coopmat()
	m := &Mixture{d: d, dim: dim, ffn: ffn, dense: dense, experts: experts, act: act, coop: coop,
		byExpertFrom: byExpertWidth()}

	push := uint32(unsafe.Sizeof(moePush{}))
	var err error
	pipes := []struct {
		into     **Pipeline
		spirv    []byte
		bindings int
	}{
		{&m.gateUp, moeGateUpSPIRV, 7},
		{&m.denseDown, matvecSPIRV, 4},
		{&m.activate, moeActSPIRV, 4},
		{&m.reduce, matmulReduceSPIRV, 2},
	}
	if experts > 0 {
		idProductSPIRV, idProductWave := matmulByIDSPIRV, uint32(0)
		if coop {
			idProductSPIRV, idProductWave = matmulCoopByIDSPIRV, coopmatWave
		}
		for _, spec := range []struct {
			into     **Pipeline
			spirv    []byte
			bindings int
			wave     uint32
		}{
			{&m.down, moeDownSPIRV, 6, 0},
			{&m.idProduct, idProductSPIRV, 7, idProductWave},
			{&m.scatter, moeScatterSPIRV, 4, 0},
			{&m.idCombine, moeIDCombineSPIRV, 3, 0},
		} {
			if *spec.into, err = d.newPipeline(spec.spirv, spec.bindings, push, spec.wave, nil); err != nil {
				m.Close()
				return nil, err
			}
		}
	}
	for _, spec := range pipes {
		if *spec.into, err = d.NewPipeline(spec.spirv, spec.bindings, push); err != nil {
			m.Close()
			return nil, err
		}
	}

	// The shared branch reads its weights once for a whole batch of positions
	// when there is one.
	for _, spec := range []struct {
		pipe    **Pipeline
		columns int
		spirv   []byte
	}{
		{&m.gateUp, smallColumns, moeGateUpWideSPIRV},
		{&m.gateUp, 16, moeGateUpMidSPIRV},
		{&m.gateUp, 32, moeGateUpWidestSPIRV},
	} {
		if err := (*spec.pipe).Wide(spec.columns, spec.spirv); err != nil {
			m.Close()
			return nil, err
		}
	}
	if coop {
		if err := m.denseDown.Wide(smallColumns, matvecWideSPIRV); err != nil {
			m.Close()
			return nil, err
		}
		for _, spec := range []struct {
			columns int
			spirv   []byte
		}{
			{tiledColumns, matmulCoop32SPIRV},
			{64, matmulCoop64SPIRV},
			{128, matmulCoop128SPIRV},
			{256, matmulCoop256SPIRV},
			{wideColumns, matmulCoop512SPIRV},
		} {
			if err := m.denseDown.WideWave(spec.columns, spec.spirv, coopmatWave); err != nil {
				m.Close()
				return nil, err
			}
		}
	} else {
		for _, spec := range []struct {
			columns int
			spirv   []byte
		}{
			{smallColumns, matvecWideSPIRV},
			{tiledColumns, matmulWide32SPIRV},
			{64, matmulWide64SPIRV},
			{128, matmulWidest128SPIRV},
			{256, matmulWidest256SPIRV},
			{wideColumns, matmulWide()},
		} {
			if err := m.denseDown.Wide(spec.columns, spec.spirv); err != nil {
				m.Close()
				return nil, err
			}
		}
	}

	// One split for every width the down projection is dispatched at, taken
	// at the widest for the reason vk/attention.go gives.
	m.splitDown = coopSplit(dim, wideColumns, coop)
	if m.gelu, err = d.Upload(asBytes(nn.GELUTableData())); err != nil {
		m.Close()
		return nil, err
	}
	if m.zero, err = d.Upload(make([]byte, 4)); err != nil {
		m.Close()
		return nil, err
	}

	in := dim / nn.QuantBlock
	dmid := dense / nn.QuantBlock
	type bufSpec struct {
		into  **Buffer
		size  int
		local bool
	}
	gateOutSize := 2 * dense * 4 * maxColumns
	if experts > 0 && 2*ffn*4*maxColumns*used > gateOutSize {
		gateOutSize = 2 * ffn * 4 * maxColumns * used
	}
	bufs := []bufSpec{
		{&m.dxq, dim * maxColumns, true},        // the shared branch's input
		{&m.dxs, 2 * in * 4 * maxColumns, true}, //
		{&m.dout, dim * 4 * maxColumns, true},   // its output
		{&m.doutParts, dim * 4 * maxColumns * matmulSplit(dim), true},
		{&m.daq, dense * maxColumns, true},        // its intermediate
		{&m.das, 2 * dmid * 4 * maxColumns, true}, //
		// Both halves of the shared branch's or expert branch's first projection,
		// in float, taking the larger of the two sizes.
		{&m.gateOut, gateOutSize, true},
	}
	if experts > 0 {
		// Every column of a pass routes for itself, and the buffers between
		// the two halves are one row per pair of a column and one of its
		// eight slots. A hundred and twenty-eight columns is a thousand and
		// twenty-four rows: 720 kilobytes of intermediate and ten megabytes
		// of down projection, against the twelve gigabytes the pass saves
		// reading the stack once instead of once a column.
		mid := ffn / nn.QuantBlock
		pairs := maxColumns * expertsUsed
		bufs = append(bufs,
			bufSpec{&m.xq, dim * maxColumns, true},        // the expert branch's input
			bufSpec{&m.xs, 2 * in * 4 * maxColumns, true}, // its scales, then its corrections
			// The router picks these and the two expert kernels read them,
			// so they are device memory like everything else between two
			// kernels: they were host-visible from the days the CPU did the
			// routing, and a shader reading system memory reaches across the
			// bus. See vk/device.go's Host, and s.xs in vk/stack.go.
			bufSpec{&m.ids, pairs * 4, true},                  // the chosen experts, a column at a time
			bufSpec{&m.cw, pairs * 4, true},                   // routing weight times expert scale
			bufSpec{&m.out, dim * 4 * maxColumns, true},       // that branch's output
			bufSpec{&m.aq, pairs * ffn, true},                 // its intermediate, a row per pair
			bufSpec{&m.as, 2 * pairs * mid * 4, true},         // and that intermediate's scales
			bufSpec{&m.counts, experts * 4, true},             // columns that chose each expert
			bufSpec{&m.pairs, experts * maxColumns * 4, true}, // and which
			bufSpec{&m.plan, (1 + 2*idPlanMax(experts, maxColumns, idProductBN)) * 4, true},
			bufSpec{&m.dpart, pairs * dim * 4, true}, // the down projection, a row per pair
		)
	}
	for _, spec := range bufs {
		var b *Buffer
		if spec.local {
			b, err = d.Local(spec.size, bufferUsageStorage)
		} else {
			b, err = d.Host(spec.size, bufferUsageStorage)
		}
		if err != nil {
			m.Close()
			return nil, err
		}
		*spec.into = b
	}
	if m.splitDown > 1 {
		if m.reduceSet, err = m.reduce.NewSet([]*Buffer{m.doutParts, m.dout}); err != nil {
			m.Close()
			return nil, err
		}
	}
	// The activation reads no weight, so one set serves every block.
	if m.actSet, err = m.activate.NewSet([]*Buffer{m.gateOut, m.gelu, m.daq, m.das}); err != nil {
		m.Close()
		return nil, err
	}
	if experts > 0 {
		if m.idActivate, err = d.NewPipeline(moeActByIDSPIRV, 4, push); err != nil {
			m.Close()
			return nil, err
		}
		if m.idActSet, err = m.idActivate.NewSet([]*Buffer{m.gateOut, m.gelu, m.aq, m.as}); err != nil {
			m.Close()
			return nil, err
		}
		// Neither of these reads a weight, so one set serves every block.
		if m.scatterSet, err = m.scatter.NewSet([]*Buffer{m.ids, m.counts, m.pairs, m.plan}); err != nil {
			m.Close()
			return nil, err
		}
		if m.combineSet, err = m.idCombine.NewSet([]*Buffer{m.dpart, m.cw, m.out}); err != nil {
			m.Close()
			return nil, err
		}
	}
	return m, nil
}

// Blocks is how many have been added.
func (m *Mixture) Blocks() int { return len(m.blocks) }

// AddBlock uploads one block's matrices, in the file's own layout, and binds
// the kernels to them. gate and up are the shared branch's two halves, which
// are concatenated here into the one matrix of twice the width the kernel
// reads. The blocks are read back in the order they were added.
// A dense checkpoint has no expert stacks: gateUpExps and downExps are nil,
// and the block is the shared branch alone.
func (m *Mixture) AddBlock(gateUpExps, downExps, gate, up, down []byte) error {
	shapes := []struct {
		what       string
		data       []byte
		rows, cols int
	}{
		{"the shared gate", gate, m.dense, m.dim},
		{"the shared up", up, m.dense, m.dim},
		{"the shared down", down, m.dim, m.dense},
	}
	if m.experts > 0 {
		shapes = append(shapes,
			struct {
				what       string
				data       []byte
				rows, cols int
			}{"the gate-and-up stack", gateUpExps, m.experts * 2 * m.ffn, m.dim},
			struct {
				what       string
				data       []byte
				rows, cols int
			}{"the down stack", downExps, m.experts * m.dim, m.ffn},
		)
	} else if gateUpExps != nil || downExps != nil {
		return fmt.Errorf("vk: this mixture was opened without experts, and the block brings %d bytes of them", len(gateUpExps)+len(downExps))
	}
	for _, spec := range shapes {
		if want := spec.rows * rowBytesQ4_0(spec.cols); len(spec.data) != want {
			return fmt.Errorf("vk: %s should be %d bytes, given %d", spec.what, want, len(spec.data))
		}
	}

	b := &mixtureBlock{}
	fail := func(err error) error {
		b.close()
		return err
	}
	var err error
	if m.experts > 0 {
		if b.gateUp, err = m.d.Upload(splitQ4_0(gateUpExps, m.experts*2*m.ffn, m.dim)); err != nil {
			return fail(err)
		}
		if b.down, err = m.d.Upload(splitQ4_0(downExps, m.experts*m.dim, m.ffn)); err != nil {
			return fail(err)
		}
	}
	// The shared branch's gate and up, one after the other, which is the
	// layout the mixture's own stack already has.
	joined := make([]byte, 0, len(gate)+len(up))
	joined = append(append(joined, gate...), up...)
	if b.denseGateUp, err = m.d.Upload(splitQ4_0(joined, 2*m.dense, m.dim)); err != nil {
		return fail(err)
	}
	if b.denseDown, err = m.d.Upload(splitQ4_0(down, m.dim, m.dense)); err != nil {
		return fail(err)
	}

	type setSpec struct {
		into *(*Set)
		pipe *Pipeline
		bufs []*Buffer
	}
	sets := []setSpec{
		{&b.setDenseUp, m.gateUp, []*Buffer{b.denseGateUp, m.dxq, m.dxs, m.zero, m.gelu, m.daq, m.das}},
		// The same weights through the tiled product, which is the wide
		// pass's path: same layout on the card, so the matrix is uploaded
		// once and the two kernels read it the same way.
		{&b.setDenseProd, m.denseDown, []*Buffer{b.denseGateUp, m.dxq, m.dxs, m.gateOut}},
		{&b.setDenseDn, m.denseDown, []*Buffer{b.denseDown, m.daq, m.das, m.dout}},
	}
	if m.splitDown > 1 {
		sets = append(sets, setSpec{&b.setDenseDnParts, m.denseDown,
			[]*Buffer{b.denseDown, m.daq, m.das, m.doutParts}})
	}
	if m.experts > 0 {
		sets = append(sets,
			setSpec{&b.setGateUp, m.gateUp, []*Buffer{b.gateUp, m.xq, m.xs, m.ids, m.gelu, m.aq, m.as}},
			setSpec{&b.setDown, m.down, []*Buffer{b.down, m.aq, m.as, m.ids, m.cw, m.out}},
			setSpec{&b.setIDProd, m.idProduct,
				[]*Buffer{b.gateUp, m.xq, m.xs, m.gateOut, m.counts, m.pairs, m.plan}},
			setSpec{&b.setIDDown, m.idProduct,
				[]*Buffer{b.down, m.aq, m.as, m.dpart, m.counts, m.pairs, m.plan}},
		)
	}
	for _, spec := range sets {
		if *spec.into, err = spec.pipe.NewSet(spec.bufs); err != nil {
			return fail(err)
		}
	}
	m.blocks = append(m.blocks, b)
	return nil
}

// Profile is Stack.Profile, forwarded: one stamp between the two halves.
func (m *Mixture) Profile(t *Timeline) { m.tl = t }

// mark names a span inside the block, and is nothing at all when no timeline
// is set. A stamp only reads the clock, but a span is only honest if what it
// measures has finished, so the barrier that pins it down goes in here — and
// stays out of the recording the rest of the time, where the two branches are
// meant to run over each other.
func (m *Mixture) mark(r *Recorder, label string) {
	if m.tl == nil {
		return
	}
	r.Barrier()
	m.tl.Stamp(r, label)
}

// RecordSharedUp and RecordSharedDown issue the shared branch's two products
// on their own, ahead of the rest of the block, and say whether they did.
//
// The shared branch of a mixture block reads what the residual kernel wrote
// and nothing the routing writes — stack.go binds its quantized input to the
// same dispatch that writes the residual. The routing beside it is three
// dispatches of almost nothing: a norm, a hundred and twenty-eight logits, and
// a sort of a hundred and twenty-eight numbers. Serialized behind their own
// barriers those cost six and a half microseconds each, which is what a
// dispatch of one workgroup costs to traverse the pipeline whatever is in it —
// measured, by serializing ten copies of the attention norm and reading the
// slope.
//
// So they are issued as passengers instead. The gate-and-up product carries
// the router's norm, the down product carries the router's logits, and the
// card runs each pair together because nothing in either reads the other. The
// arithmetic is untouched: the same kernels read the same bytes and write the
// same answers, and only what they ride beside changes.
//
// One column only. A prompt's shared branch is a tiled product with a barrier
// inside it, and the prompt is not the side that needs this.
func (m *Mixture) RecordSharedUp(r *Recorder, block, columns int) bool {
	if m.blocks[block].golem != nil || passWidth(columns) != 1 {
		// A Golem branch is one recording, not two halves with the experts
		// between them: there are no experts to put there.
		return false
	}
	shared := moePush{dim: uint32(m.dim), ffn: uint32(m.dense), used: 1, act: uint32(m.act), split: 1}
	r.Dispatch(m.blocks[block].setDenseUp, uint32(m.dense/nn.QuantBlock), unsafe.Pointer(&shared))
	return true
}

func (m *Mixture) RecordSharedDown(r *Recorder, block, columns int) bool {
	if m.blocks[block].golem != nil || passWidth(columns) != 1 {
		return false
	}
	shared := moePush{dim: uint32(m.dim), ffn: uint32(m.dense), used: 1, act: uint32(m.act), split: 1}
	r.Dispatch(m.blocks[block].setDenseDn, m.productGroups(1, m.dim), unsafe.Pointer(&shared))
	return true
}

// Record puts one block's two branches into a recording without submitting it,
// which is what running a whole token in one submission needs. The inputs and
// the routing must already be in the buffers the accessors below name, and
// columns is the width of the pass: more than one is a stretch of a prompt,
// and only the shared branch can take it — see the note beside the wide
// pipelines above.
//
// sharedUp and sharedDown say those two products have already been issued by
// the two calls above, and this recording must not issue them twice.
func (m *Mixture) Record(r *Recorder, block, columns int, sharedUp, sharedDown bool) {
	if b := m.blocks[block]; b.golem != nil {
		if err := m.recordGolem(r, block, columns); err != nil {
			panic(fmt.Sprintf("vk: the Golem feed forward refused a pass of %d: %v", columns, err))
		}
		return
	}
	experts := moePush{dim: uint32(m.dim), ffn: uint32(m.ffn), used: expertsUsed, act: uint32(m.act),
		cap: uint32(maxColumns), split: 1}
	shared := moePush{dim: uint32(m.dim), ffn: uint32(m.dense), used: 1, act: uint32(m.act), split: 1}
	b := m.blocks[block]
	width := passWidth(columns)
	up, down := uint32(m.dense/nn.QuantBlock), m.productGroups(width, m.dim)
	// Nothing in either branch waits on the other, so they go in without a
	// barrier between them and the card runs them together.
	// A token reads the expert stack a column at a time — eight matrices for
	// one position, and nothing between two positions to share. A prompt reads
	// it by expert instead: shaders/moe_scatter.comp turns the routing inside
	// out, and the two halves below walk one expert's list of columns rather
	// than one column's list of experts. Same answer, and the stack read once
	// for a pass instead of once for each of its columns.
	//
	// Which is cheaper depends on the width, and byExpertFrom says where they
	// cross. This used to read `columns > 1`, which made a pass of two columns
	// the worst of both: a hundred and twenty-eight experts read to answer
	// sixteen pairs. A server drawing a token for each of two conversations
	// lives at exactly that width.
	byExpert := m.experts > 0 && columns >= m.byExpertFrom
	bn := uint32(idBN(m.coop))
	if byExpert {
		scat := scatterPush{columns: uint32(columns), used: expertsUsed,
			experts: uint32(m.experts), cap: uint32(maxColumns), bn: bn}
		r.Dispatch(m.scatterSet, 1, unsafe.Pointer(&scat))
		r.Barrier()
		m.tl.Stamp(r, "moe scatter")

		// The gate and up halves as one product of 2*ffn rows, then the
		// activation over what it wrote. Two kernels where there was one, and
		// shaders/moe_act.comp's header says what that trade buys: the fused
		// kernel kept an accumulator a column in registers, so it carried eight
		// columns and read the expert stack once per eight rather than once.
		prod := moePush{dim: uint32(2 * m.ffn), ffn: uint32(m.dim), used: expertsUsed,
			act: uint32(m.act), bycol: 1, cap: uint32(maxColumns), split: 1}
		prodRows := uint32((2*m.ffn + matmulRows - 1) / matmulRows)
		if m.coop {
			prodRows = uint32((2*m.ffn + idProductCoopBM - 1) / idProductCoopBM)
		}
		r.Dispatch(b.setIDProd, prodRows*uint32(idPlanMax(m.experts, columns, idBN(m.coop))), unsafe.Pointer(&prod))
		r.Barrier()
		m.tl.Stamp(r, "moe expert gate/up")

		act := moePush{dim: uint32(2 * m.ffn), ffn: uint32(m.ffn), used: expertsUsed,
			act: uint32(m.act), cap: uint32(maxColumns)}
		r.DispatchColumns(m.idActSet, uint32((m.ffn+255)/256), uint32(columns*expertsUsed), unsafe.Pointer(&act))
		m.mark(r, "moe expert act")
	} else if m.experts > 0 {
		// A dispatch a column, each reading that column's own eight matrices.
		// The columns write disjoint rows of the intermediate, so nothing
		// between them has to wait.
		for c := 0; c < columns; c++ {
			at := experts
			at.col = uint32(c)
			r.Dispatch(b.setGateUp, uint32(expertsUsed*m.ffn/nn.QuantBlock), unsafe.Pointer(&at))
		}
	}
	switch {
	case width >= tiledColumns:
		// The tiled product and then the activation, which is two kernels
		// where the fused one was a kernel. shaders/moe_act.comp says why:
		// the fused kernel carries eight columns because that is what a wave
		// holds, so a pass of two hundred and fifty-six ran it thirty-two
		// times and read the gate and up matrices thirty-two times with it.
		product := moePush{dim: uint32(2 * m.dense), ffn: uint32(m.dim), used: 1, split: 1}
		r.DispatchWide(b.setDenseProd, width, m.productGroups(width, 2*m.dense), unsafe.Pointer(&product))
		r.Barrier()
		gelu := moePush{dim: uint32(2 * m.dense), ffn: uint32(m.dense), used: 1, act: uint32(m.act)}
		r.DispatchColumns(m.actSet, uint32((m.dense+255)/256), uint32(width), unsafe.Pointer(&gelu))
	case width > 1:
		gate := gateColumns
		if width < gate {
			gate = width
		}
		for c := 0; c < width; c += gate {
			at := shared
			at.col = uint32(c)
			r.DispatchWide(b.setDenseUp, gate, up, unsafe.Pointer(&at))
		}
	default:
		if !sharedUp {
			r.Dispatch(b.setDenseUp, up, unsafe.Pointer(&shared))
		}
	}
	r.Barrier()
	m.tl.Stamp(r, "moe shared gate/up")
	if byExpert {
		downPush := moePush{dim: uint32(m.dim), ffn: uint32(m.ffn), used: expertsUsed,
			act: uint32(m.act), bycol: 0, cap: uint32(maxColumns), split: 1}
		rows := uint32((m.dim + matmulRows - 1) / matmulRows)
		if m.coop {
			rows = uint32((m.dim + idProductCoopBM - 1) / idProductCoopBM)
		}
		r.Dispatch(b.setIDDown, rows*uint32(idPlanMax(m.experts, columns, idBN(m.coop))), unsafe.Pointer(&downPush))
		m.mark(r, "moe expert down")
	} else if m.experts > 0 {
		for c := 0; c < columns; c++ {
			at := experts
			at.col = uint32(c)
			r.Dispatch(b.setDown, uint32(m.dim/downOuts), unsafe.Pointer(&at))
		}
	}
	if b.setDenseDnParts != nil && width >= tiledColumns {
		split := shared
		split.split = uint32(m.splitDown)
		r.DispatchWide(b.setDenseDnParts, width, down*uint32(m.splitDown), unsafe.Pointer(&split))
		r.Barrier()
		fold := moePush{dim: uint32(m.dim), ffn: uint32(width), used: uint32(m.splitDown)}
		r.Dispatch(m.reduceSet, uint32((m.dim*width+255)/256), unsafe.Pointer(&fold))
	} else if width > 1 {
		r.DispatchWide(b.setDenseDn, width, down, unsafe.Pointer(&shared))
	} else if !sharedDown {
		r.Dispatch(b.setDenseDn, down, unsafe.Pointer(&shared))
	}
	r.Barrier()
	m.tl.Stamp(r, "moe shared down")
	if byExpert {
		// The eight rows a column's experts wrote, weighted and added. It
		// waits on the down projection above and on nothing else, so it goes
		// in after the shared branch rather than between the two.
		fold := combinePairsPush{dim: uint32(m.dim), used: expertsUsed, columns: uint32(columns)}
		r.Dispatch(m.combineSet, uint32((m.dim*columns+255)/256), unsafe.Pointer(&fold))
		m.mark(r, "moe combine pairs")
	}
}

// The buffers a kernel upstream writes and one downstream reads, so that a
// whole token can be recorded without anything crossing the bus.

// ExpertInput is where the expert branch reads its normed input, Q8_0.
func (m *Mixture) ExpertInput() (*Buffer, *Buffer) { return m.xq, m.xs }

// SharedInput is the same for the branch beside it, which reads the residual
// under a different norm.
func (m *Mixture) SharedInput() (*Buffer, *Buffer) { return m.dxq, m.dxs }

// Routing is the chosen experts and their weights, which the router writes.
func (m *Mixture) Routing() (*Buffer, *Buffer) { return m.ids, m.cw }

// Outputs are the two branches' answers, unnormed and unadded.
func (m *Mixture) Outputs() (shared, experts *Buffer) { return m.dout, m.out }

func (m *Mixture) Close() {
	if m.dxf != nil {
		m.dxf.Close()
		m.dxf = nil
	}
	for _, b := range m.blocks {
		if b.golem != nil {
			b.golem.Close()
			b.golem = nil
		}
	}
	for _, b := range m.blocks {
		b.close()
	}
	m.blocks = nil
	for _, b := range []**Buffer{
		&m.gateOut, &m.das, &m.daq, &m.doutParts, &m.dout, &m.dxs, &m.dxq,
		&m.as, &m.aq, &m.out, &m.cw, &m.ids, &m.xs, &m.xq,
		&m.dpart, &m.plan, &m.pairs, &m.counts,
		&m.zero, &m.gelu,
	} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
	for _, s := range []**Set{&m.reduceSet, &m.actSet, &m.idActSet, &m.scatterSet, &m.combineSet} {
		if *s != nil {
			(*s).Close()
			*s = nil
		}
	}
	for _, p := range []**Pipeline{
		&m.denseDown, &m.down, &m.gateUp, &m.activate, &m.idActivate, &m.reduce,
		&m.idProduct, &m.scatter, &m.idCombine,
	} {
		if *p != nil {
			(*p).Close()
			*p = nil
		}
	}
}

func (b *mixtureBlock) close() {
	for _, s := range []**Set{
		&b.setDenseDn, &b.setDenseDnParts, &b.setDenseUp, &b.setDenseProd,
		&b.setDown, &b.setGateUp, &b.setIDProd, &b.setIDDown,
	} {
		if *s != nil {
			(*s).Close()
			*s = nil
		}
	}
	for _, x := range []**Buffer{&b.denseDown, &b.denseGateUp, &b.down, &b.gateUp} {
		if *x != nil {
			(*x).Close()
			*x = nil
		}
	}
}

// rowBytesQ4_0 is what one row of that many inputs occupies.
func rowBytesQ4_0(cols int) int { return cols / nn.QuantBlock * 18 }

// splitQ4_0 rewrites rows so that a shader can reach them.
//
// A Q4_0 block is an fp16 scale followed by sixteen bytes of nibbles, and
// eighteen is not a multiple of four: every other block of a row would begin
// off alignment, and a shader indexing a uint array would pay for it on every
// load. So a row is split in two — all of its scales, then all of its nibbles
// — which is the same bytes in the same number, with both halves aligned. The
// block count is even on every shape this reads, so the nibbles start aligned
// too.
func splitQ4_0(src []byte, rows, cols int) []byte {
	nb := cols / nn.QuantBlock
	stride := nb * 18
	dst := make([]byte, len(src))
	for r := 0; r < rows; r++ {
		in := src[r*stride : (r+1)*stride]
		out := dst[r*stride : (r+1)*stride]
		nibbles := out[2*nb:]
		for b := 0; b < nb; b++ {
			block := in[b*18 : (b+1)*18]
			binary.LittleEndian.PutUint16(out[2*b:], binary.LittleEndian.Uint16(block))
			copy(nibbles[b*16:], block[2:])
		}
	}
	return dst
}

// splitQ4_1 is splitQ4_0 for the format with a minimum beside the scale: the
// twenty bytes of a block are a scale, a minimum, and sixteen nibble pairs, and
// the kernels want all the scale-and-minimum pairs of a row before any of its
// nibbles.
func splitQ4_1(src []byte, rows, cols int) []byte {
	nb := cols / nn.QuantBlock
	stride := nb * 20
	dst := make([]byte, len(src))
	for r := 0; r < rows; r++ {
		in := src[r*stride : (r+1)*stride]
		out := dst[r*stride : (r+1)*stride]
		nibbles := out[4*nb:]
		for b := 0; b < nb; b++ {
			block := in[b*20 : (b+1)*20]
			copy(out[4*b:], block[:4])
			copy(nibbles[b*16:], block[4:])
		}
	}
	return dst
}

// asBytes views a float slice as the bytes behind it, for an upload.
func asBytes(f []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), len(f)*4)
}

func (m *Mixture) productGroups(width, outputs int) uint32 {
	return coopProductGroups(m.coop, width, outputs)
}

// rowBytesQ5_K is what one row of that many inputs occupies once splitQ5_K has
// had it: twenty-four bytes to a block of thirty-two, against the format's own
// twenty-two.
func rowBytesQ5_K(cols int) int { return cols / nn.QuantBlock * 24 }

// splitQ5_K rewrites a Q5_K matrix into blocks of thirty-two that stand alone.
//
// This is the one format here whose blocks do not: a Q5_K superblock is two
// fp16 magnitudes, twelve bytes of six-bit scales and minima for its eight
// sub-blocks, thirty-two bytes of fifth bits, and a hundred and twenty-eight
// of nibbles — and the fifth bits are *bit-sliced*, so sub-block i is bit i of
// all thirty-two of those bytes. A kernel that wants one sub-block has to read
// all thirty-two, and the header on top of that. Staged a sub-block to a
// thread, which is the shape every other product here has, that is eighty
// bytes read where twenty-two are wanted.
//
// That is why the delta net's output projection was the last matrix in the
// model still on a hand-rolled tile at a quarter of the cooperative kernel's
// throughput: the cooperative kernel stages a block to a thread and there was
// no way to give it one. So the block is made to stand alone here instead, on
// the way to the card, once:
//
//	[0,      4nb)  d1 and m1, one fp16 pair a block, already multiplied out
//	[4nb,    8nb)  the fifth bits, one uint a block, bit l for weight l
//	[8nb,   24nb)  the nibbles, sixteen bytes a block, in Q4_0's own order —
//	               low nibble of byte j is weight j, high nibble is weight j+16
//
// Six bits a weight where the format spends five and a half, and in exchange
// the staging is Q4_0's with three more instructions: spread four bits over
// four bytes, or them into the nibbles, and subtract a minimum instead of
// eight. shaders/matmul_coop.comp reads this under -DQ5K.
//
// The scale and minimum are folded here rather than staged, which the format
// does not lose anything by: d*sc[i] and dmin*m[i] are what every reader
// computed anyway, and an fp16 pair a block is what Q4_1 already carries.
func splitQ5_K(src []byte, rows, cols int) []byte {
	const superBytes = 176
	nb := cols / nn.QuantBlock // blocks of thirty-two
	nsb := cols / 256          // superblocks
	inStride := nsb * superBytes
	outStride := nb * 24
	dst := make([]byte, rows*outStride)
	for r := 0; r < rows; r++ {
		in := src[r*inStride : (r+1)*inStride]
		out := dst[r*outStride : (r+1)*outStride]
		highs := out[4*nb:]
		nibbles := out[8*nb:]
		for sb := 0; sb < nsb; sb++ {
			block := in[sb*superBytes : (sb+1)*superBytes]
			d := nn.Widen(binary.LittleEndian.Uint16(block[0:]))
			dmin := nn.Widen(binary.LittleEndian.Uint16(block[2:]))
			raw := block[4:16]
			qh := block[16:48]
			qs := block[48:176]

			var sc, m [8]uint8
			for i := 0; i < 4; i++ {
				sc[i] = raw[i] & 63
				m[i] = raw[i+4] & 63
				sc[i+4] = (raw[i+8] & 0xF) | ((raw[i] >> 6) << 4)
				m[i+4] = (raw[i+8] >> 4) | ((raw[i+4] >> 6) << 4)
			}

			for i := 0; i < 8; i++ {
				b := sb*8 + i
				d1 := d * float32(sc[i])
				m1 := dmin * float32(m[i])
				binary.LittleEndian.PutUint16(out[4*b:], nn.Narrow(d1))
				binary.LittleEndian.PutUint16(out[4*b+2:], nn.Narrow(m1))

				qSub := qs[(i/2)*32:]
				shift := uint(4 * (i & 1))
				qhBit := uint8(1) << uint(i)

				// Q4_0's order: byte j carries weight j low and weight j+16
				// high, so that the staging can mask 0x0F0F0F0F and shift by
				// four exactly as it does for a Q4_0 block.
				var hi uint32
				pack := nibbles[b*16 : b*16+16]
				for j := 0; j < 16; j++ {
					lo := (qSub[j] >> shift) & 0xF
					up := (qSub[j+16] >> shift) & 0xF
					pack[j] = lo | up<<4
					if qh[j]&qhBit != 0 {
						hi |= 1 << uint(j)
					}
					if qh[j+16]&qhBit != 0 {
						hi |= 1 << uint(j+16)
					}
				}
				binary.LittleEndian.PutUint32(highs[4*b:], hi)
			}
		}
	}
	return dst
}
