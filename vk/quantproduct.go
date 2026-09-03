package vk

// One weight format, one pipeline, and the same four buffers whichever it is.
//
// Every projection on this card is the same product: a quantized matrix against
// the Q8_0 form of whatever feeds it, answering a float. What differs between
// Q4_0, Q4_1, Q4_K, Q5_K and Q6_K is thirty lines of unpacking inside two
// shaders — the mat-vec a token takes and the tiled product a prompt takes —
// and nothing outside them. So a caller that has a matrix in one of these
// formats asks for its pipeline and binds the same set it always did.
//
// That is the whole of what an engine needs to read llama.cpp's quantized
// forms, and it is why this file is the door rather than each engine's own
// wiring: vk/qwen_pipeline.go carried seven pipelines and four booleans to say
// which of them a matrix wanted, and every one of those booleans had a default
// that was a guess about somebody else's file.
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
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ4K --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q4kq8.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ4K -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q4kq8_2.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ4K -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q4kq8_4.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ4K -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q4kq8_8.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ4K -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q4kq8_16.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ6K --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q6kq8.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ6K -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q6kq8_2.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ6K -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q6kq8_4.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ6K -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q6kq8_8.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ6K -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q6kq8_16.spv
// And Q3_K, which nothing here writes and which is read because a client who
// will not compress a checkpoint downloads one. vk/split_q3k.go is its layout.
//
// The workgroup shape is written out here, as it now is on every mat-vec
// directive that is not Q8_0's. It was written on none of them: shaders/ held
// binaries built at sixteen lanes to a row while the directives asked for the
// source's default of eight, so `go generate` rewrote thirty shaders at a
// geometry nobody had measured, and it did it silently — a mat-vec at the wrong
// lane count answers, more slowly. The Q8_0 family really is at the default and
// says so by carrying no flag.
//
//go:generate glslc -O -DQ3K -DLANES=16 -DOUTS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q3kq8.spv
//go:generate glslc -O -DQ3K -DLANES=16 -DOUTS=8 -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q3kq8_2.spv
//go:generate glslc -O -DQ3K -DLANES=16 -DOUTS=8 -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q3kq8_4.spv
//go:generate glslc -O -DQ3K -DLANES=16 -DOUTS=8 -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q3kq8_8.spv
//go:generate glslc -O -DQ3K -DLANES=16 -DOUTS=8 -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q3kq8_16.spv
//go:generate glslc -O -DQ3K -DCOLUMNS=32 -DBN=32 -DBM=64 -DBK=128 -DWAVE_M=4 -DWAVE_N=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q3k32.spv
//go:generate glslc -O -DQ3K -DCOLUMNS=64 -DBN=64 -DBM=128 -DBK=128 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q3k64.spv
//go:generate glslc -O -DQ3K -DCOLUMNS=128 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q3k128.spv
//go:generate glslc -O -DQ3K -DCOLUMNS=256 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q3k256.spv
//go:generate glslc -O -DQ3K -DCOLUMNS=512 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q3k512.spv

// And Q2_K, the tier below it, which llama.cpp's two-bit mix puts on the
// query, the key, the gate and the up. vk/split_q2k.go is its layout.
//
//go:generate glslc -O -DQ2K -DLANES=16 -DOUTS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q2kq8.spv
//go:generate glslc -O -DQ2K -DLANES=16 -DOUTS=8 -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q2kq8_2.spv
//go:generate glslc -O -DQ2K -DLANES=16 -DOUTS=8 -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q2kq8_4.spv
//go:generate glslc -O -DQ2K -DLANES=16 -DOUTS=8 -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q2kq8_8.spv
//go:generate glslc -O -DQ2K -DLANES=16 -DOUTS=8 -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q2kq8_16.spv
//go:generate glslc -O -DQ2K -DCOLUMNS=32 -DBN=32 -DBM=64 -DBK=128 -DWAVE_M=4 -DWAVE_N=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q2k32.spv
//go:generate glslc -O -DQ2K -DCOLUMNS=64 -DBN=64 -DBM=128 -DBK=128 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q2k64.spv
//go:generate glslc -O -DQ2K -DCOLUMNS=128 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q2k128.spv
//go:generate glslc -O -DQ2K -DCOLUMNS=256 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q2k256.spv
//go:generate glslc -O -DQ2K -DCOLUMNS=512 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q2k512.spv

//go:generate glslc -O -DQ4K -DCOLUMNS=32 -DBN=32 -DBM=64 -DBK=128 -DWAVE_M=4 -DWAVE_N=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q4k32.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=32 -DBN=32 -DBM=64 -DBK=128 -DWAVE_M=4 -DWAVE_N=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q6k32.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=64 -DBN=64 -DBM=128 -DBK=128 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q6k64.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=128 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q6k128.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=256 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q6k256.spv
//go:generate glslc -O -DQ6K -DCOLUMNS=512 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q6k512.spv

// And the same two shaders over the two formats that had a kernel of their own
// before this file existed. Q4_1 is what golem's own converter writes for the
// blocks a Q4_0 loses too much on; Q5_K is what llama.cpp puts on a delta net's
// output projection. Both were read against *float* activations by kernels
// that predate the packed dot, which is a slower product for the same answer.
//
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ41 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q41q8.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ41 -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q41q8_2.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ41 -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q41q8_4.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ41 -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q41q8_8.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ41 -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q41q8_16.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ5K --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q5kq8.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ5K -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q5kq8_2.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ5K -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q5kq8_4.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ5K -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q5kq8_8.spv
//go:generate glslc -O -DLANES=16 -DOUTS=8 -DQ5K -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q5kq8_16.spv
//go:generate glslc -O -DQ5K -DCOLUMNS=32 -DBN=32 -DBM=64 -DBK=128 -DWAVE_M=4 -DWAVE_N=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q5k32.spv
//go:generate glslc -O -DQ41 -DCOLUMNS=32 -DBN=32 -DBM=64 -DBK=128 -DWAVE_M=4 -DWAVE_N=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q41_32.spv
//go:generate glslc -O -DQ41 -DCOLUMNS=64 -DBN=64 -DBM=128 -DBK=128 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q41_64.spv
//go:generate glslc -O -DQ41 -DCOLUMNS=128 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q41_128.spv
//go:generate glslc -O -DQ41 -DCOLUMNS=256 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q41_256.spv
//go:generate glslc -O -DQ41 -DCOLUMNS=512 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q41_512.spv

// The projection kernels against Q8_0 weights. They exist for a mixture whose
// experts are Q8_0, because such a checkpoint is Q8_0 throughout: its dense
// branch and its attention come through this door as well, and refusing them
// would refuse the model the routed kernels were widened for.
//
// No tiled form. A card without matrix cores answers a wide pass sixteen
// columns at a time for every format but Q4_0 already, and that is the path
// this takes on every card — a Q8_0 checkpoint is read for its size, not for
// its prompt speed.
//
//go:generate glslc -O -DQ80 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q80w.spv
//go:generate glslc -O -DQ80 -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q80w2.spv
//go:generate glslc -O -DQ80 -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q80w4.spv
//go:generate glslc -O -DQ80 -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q80w8.spv
//go:generate glslc -O -DQ80 -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_q80w16.spv
//go:embed shaders/matvec_q80w.spv
var matvecQ80WSPIRV []byte

//go:embed shaders/matvec_q80w2.spv
var matvecQ80W2SPIRV []byte

//go:embed shaders/matvec_q80w4.spv
var matvecQ80W4SPIRV []byte

//go:embed shaders/matvec_q80w8.spv
var matvecQ80W8SPIRV []byte

//go:embed shaders/matvec_q80w16.spv
var matvecQ80W16SPIRV []byte

//go:embed shaders/matvec_q41q8.spv
var matvecQ41Q8SPIRV []byte

//go:embed shaders/matvec_q41q8_2.spv
var matvecQ41Q8_2SPIRV []byte

//go:embed shaders/matvec_q41q8_4.spv
var matvecQ41Q8_4SPIRV []byte

//go:embed shaders/matvec_q41q8_8.spv
var matvecQ41Q8_8SPIRV []byte

//go:embed shaders/matvec_q41q8_16.spv
var matvecQ41Q8_16SPIRV []byte

//go:embed shaders/matvec_q5kq8.spv
var matvecQ5KQ8SPIRV []byte

//go:embed shaders/matvec_q5kq8_2.spv
var matvecQ5KQ8_2SPIRV []byte

//go:embed shaders/matvec_q5kq8_4.spv
var matvecQ5KQ8_4SPIRV []byte

//go:embed shaders/matvec_q5kq8_8.spv
var matvecQ5KQ8_8SPIRV []byte

//go:embed shaders/matvec_q5kq8_16.spv
var matvecQ5KQ8_16SPIRV []byte

//go:embed shaders/matmul_coop_q5k32.spv
var matmulCoopQ5K32SPIRV []byte

//go:embed shaders/matmul_coop_q41_32.spv
var matmulCoopQ41_32SPIRV []byte

//go:embed shaders/matmul_coop_q41_64.spv
var matmulCoopQ41_64SPIRV []byte

//go:embed shaders/matmul_coop_q41_128.spv
var matmulCoopQ41_128SPIRV []byte

//go:embed shaders/matmul_coop_q41_256.spv
var matmulCoopQ41_256SPIRV []byte

//go:embed shaders/matmul_coop_q41_512.spv
var matmulCoopQ41_512SPIRV []byte

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

//go:embed shaders/matvec_q3kq8.spv
var matvecQ3KQ8SPIRV []byte

//go:embed shaders/matvec_q3kq8_2.spv
var matvecQ3KQ8_2SPIRV []byte

//go:embed shaders/matvec_q3kq8_4.spv
var matvecQ3KQ8_4SPIRV []byte

//go:embed shaders/matvec_q3kq8_8.spv
var matvecQ3KQ8_8SPIRV []byte

//go:embed shaders/matvec_q3kq8_16.spv
var matvecQ3KQ8_16SPIRV []byte

//go:embed shaders/matmul_coop_q3k32.spv
var matmulCoopQ3K32SPIRV []byte

//go:embed shaders/matmul_coop_q3k64.spv
var matmulCoopQ3K64SPIRV []byte

//go:embed shaders/matmul_coop_q3k128.spv
var matmulCoopQ3K128SPIRV []byte

//go:embed shaders/matmul_coop_q3k256.spv
var matmulCoopQ3K256SPIRV []byte

//go:embed shaders/matmul_coop_q3k512.spv
var matmulCoopQ3K512SPIRV []byte

//go:embed shaders/matvec_q2kq8.spv
var matvecQ2KQ8SPIRV []byte

//go:embed shaders/matvec_q2kq8_2.spv
var matvecQ2KQ8_2SPIRV []byte

//go:embed shaders/matvec_q2kq8_4.spv
var matvecQ2KQ8_4SPIRV []byte

//go:embed shaders/matvec_q2kq8_8.spv
var matvecQ2KQ8_8SPIRV []byte

//go:embed shaders/matvec_q2kq8_16.spv
var matvecQ2KQ8_16SPIRV []byte

//go:embed shaders/matmul_coop_q2k32.spv
var matmulCoopQ2K32SPIRV []byte

//go:embed shaders/matmul_coop_q2k64.spv
var matmulCoopQ2K64SPIRV []byte

//go:embed shaders/matmul_coop_q2k128.spv
var matmulCoopQ2K128SPIRV []byte

//go:embed shaders/matmul_coop_q2k256.spv
var matmulCoopQ2K256SPIRV []byte

//go:embed shaders/matmul_coop_q2k512.spv
var matmulCoopQ2K512SPIRV []byte

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
	case nn.Q4_0, nn.Q4_1, nn.Q2_K, nn.Q3_K, nn.Q4_K, nn.Q5_K, nn.Q6_K, nn.Q8_0:
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
	case nn.Q4_1:
		return cols / nn.QuantBlock * 20, nil
	case nn.Q2_K:
		return rowBytesQ2_K(cols), nil
	case nn.Q3_K:
		return rowBytesQ3_K(cols), nil
	case nn.Q4_K:
		return rowBytesQ4_K(cols), nil
	case nn.Q5_K:
		return rowBytesQ5_K(cols), nil
	case nn.Q6_K:
		return rowBytesQ6_K(cols), nil
	case nn.Q8_0:
		return cols / nn.QuantBlock * 34, nil
	}
	return 0, fmt.Errorf("vk: there is no projection kernel for %s", q)
}

// quantLayout is a matrix in the layout its kernels read, checked against the
// length the format says it should have.
func quantLayout(q nn.Quant, data []byte, rows, cols int) ([]byte, error) {
	if (q == nn.Q2_K || q == nn.Q3_K || q == nn.Q4_K || q == nn.Q5_K || q == nn.Q6_K) && cols%nn.SuperBlock != 0 {
		return nil, fmt.Errorf("vk: a %s row needs a multiple of %d columns, given %d", q, nn.SuperBlock, cols)
	}
	if cols%nn.QuantBlock != 0 {
		return nil, fmt.Errorf("vk: a quantized row needs a multiple of %d columns, given %d", nn.QuantBlock, cols)
	}
	fileRow, err := quantFileRow(q, cols)
	if err != nil {
		return nil, err
	}
	if want := rows * fileRow; len(data) != want {
		return nil, fmt.Errorf("vk: %d rows of %d columns in %s need %d bytes, given %d", rows, cols, q, want, len(data))
	}
	switch q {
	case nn.Q4_0:
		return splitQ4_0(data, rows, cols), nil
	case nn.Q4_1:
		return splitQ4_1(data, rows, cols), nil
	case nn.Q2_K:
		return splitQ2_K(data, rows, cols), nil
	case nn.Q3_K:
		return splitQ3_K(data, rows, cols), nil
	case nn.Q4_K:
		return splitQ4_K(data, rows, cols), nil
	case nn.Q5_K:
		return splitQ5_K(data, rows, cols), nil
	case nn.Q8_0:
		return splitQ8_0(data, rows, cols), nil
	default:
		return splitQ6_K(data, rows, cols), nil
	}
}

// quantFileRow is what one row of that many inputs occupies in the *file*,
// which is not always what it occupies once a packing has had it.
//
// Three of these formats differ between the two, and the difference is what
// this function exists to keep in one place. Q5_K's superblock grows from a
// hundred and seventy-six bytes to a hundred and ninety-two, Q3_K's from a
// hundred and ten to a hundred and fourteen, and Q6_K's two hundred and ten
// stay two hundred and ten but are rounded up to a word — so a Q6_K row of
// fifteen superblocks is 3150 bytes in the file and 3152 on the card.
//
// A second copy of this arithmetic in vk/matmul.go length-checked callers'
// tensors against the *packed* count, and so refused every Q6_K matrix with an
// odd number of superblocks to a row. 3840 columns is fifteen of them, which
// is Gemma 4 12B's attention width. The formats whose rows are a whole number
// of words either way — Q4_0, Q4_K, Q8_0 — hid it.
func quantFileRow(q nn.Quant, cols int) (int, error) {
	switch q {
	case nn.Q4_0:
		return rowBytesQ4_0(cols), nil
	case nn.Q4_1:
		return cols / nn.QuantBlock * 20, nil
	case nn.Q2_K:
		// Eighty-four bytes a superblock, which the packing keeps: Q2_K is the
		// one format here the card holds at exactly the file's size.
		return cols / nn.SuperBlock * 84, nil
	case nn.Q3_K:
		// A hundred and ten bytes a superblock, which the packing takes to a
		// hundred and fourteen; vk/split_q3k.go says what the four buy.
		return cols / nn.SuperBlock * 110, nil
	case nn.Q4_K:
		return rowBytesQ4_K(cols), nil
	case nn.Q5_K:
		// A hundred and seventy-six bytes a superblock, which the packing takes
		// to a hundred and ninety-two; splitQ5_K says what the sixth bit buys.
		return cols / nn.SuperBlock * 176, nil
	case nn.Q6_K:
		return cols / nn.SuperBlock * 210, nil
	case nn.Q8_0:
		return cols / nn.QuantBlock * 34, nil
	}
	return 0, fmt.Errorf("vk: there is no projection kernel for %s", q)
}

// newQuantProduct builds the pipeline that reads one weight format: the mat-vec
// at the narrow widths, the tiled product at the wide ones.
//
// Every format is built at one, two, four, eight and sixteen columns and then
// at the five tiled widths, which is what vk/qwen_pipeline.go's qwenWidths asks
// for by name: a pass takes the width it wants and there is no falling back to
// a narrower binary. On a card without matrix cores only Q4_0 has a tiled form,
// so the others answer a wide pass sixteen columns at a time — slower, and the
// alternative was refusing the model.
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
		for _, w := range []struct {
			columns int
			spirv   []byte
		}{{2, matvec2SPIRV}, {4, matvec4SPIRV}, {smallColumns, matvecWideSPIRV}, {16, matvecQwen16SPIRV}} {
			narrow = append(narrow, w)
		}
	case nn.Q2_K:
		base = matvecQ2KQ8SPIRV
		for _, w := range []struct {
			columns int
			spirv   []byte
		}{{2, matvecQ2KQ8_2SPIRV}, {4, matvecQ2KQ8_4SPIRV}, {smallColumns, matvecQ2KQ8_8SPIRV}, {16, matvecQ2KQ8_16SPIRV}} {
			narrow = append(narrow, w)
		}
	case nn.Q3_K:
		base = matvecQ3KQ8SPIRV
		for _, w := range []struct {
			columns int
			spirv   []byte
		}{{2, matvecQ3KQ8_2SPIRV}, {4, matvecQ3KQ8_4SPIRV}, {smallColumns, matvecQ3KQ8_8SPIRV}, {16, matvecQ3KQ8_16SPIRV}} {
			narrow = append(narrow, w)
		}
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
	case nn.Q4_1:
		base = matvecQ41Q8SPIRV
		for _, w := range []struct {
			columns int
			spirv   []byte
		}{{2, matvecQ41Q8_2SPIRV}, {4, matvecQ41Q8_4SPIRV}, {smallColumns, matvecQ41Q8_8SPIRV}, {16, matvecQ41Q8_16SPIRV}} {
			narrow = append(narrow, w)
		}
	case nn.Q5_K:
		base = matvecQ5KQ8SPIRV
		for _, w := range []struct {
			columns int
			spirv   []byte
		}{{2, matvecQ5KQ8_2SPIRV}, {4, matvecQ5KQ8_4SPIRV}, {smallColumns, matvecQ5KQ8_8SPIRV}, {16, matvecQ5KQ8_16SPIRV}} {
			narrow = append(narrow, w)
		}
	case nn.Q8_0:
		base = matvecQ80WSPIRV
		for _, w := range []struct {
			columns int
			spirv   []byte
		}{{2, matvecQ80W2SPIRV}, {4, matvecQ80W4SPIRV}, {smallColumns, matvecQ80W8SPIRV}, {16, matvecQ80W16SPIRV}} {
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
		case nn.Q2_K:
			tiled = []struct {
				columns int
				spirv   []byte
			}{
				{tiledColumns, matmulCoopQ2K32SPIRV}, {64, matmulCoopQ2K64SPIRV},
				{128, matmulCoopQ2K128SPIRV}, {256, matmulCoopQ2K256SPIRV}, {wideColumns, matmulCoopQ2K512SPIRV},
			}
		case nn.Q3_K:
			tiled = []struct {
				columns int
				spirv   []byte
			}{
				{tiledColumns, matmulCoopQ3K32SPIRV}, {64, matmulCoopQ3K64SPIRV},
				{128, matmulCoopQ3K128SPIRV}, {256, matmulCoopQ3K256SPIRV}, {wideColumns, matmulCoopQ3K512SPIRV},
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
		case nn.Q5_K:
			tiled = []struct {
				columns int
				spirv   []byte
			}{
				{tiledColumns, matmulCoopQ5K32SPIRV}, {64, matmulCoopQ5K64SPIRV},
				{128, matmulCoopQ5K128SPIRV}, {256, matmulCoopQ5K256SPIRV}, {wideColumns, matmulCoopQ5K512SPIRV},
			}
		case nn.Q4_1:
			tiled = []struct {
				columns int
				spirv   []byte
			}{
				{tiledColumns, matmulCoopQ41_32SPIRV}, {64, matmulCoopQ41_64SPIRV},
				{128, matmulCoopQ41_128SPIRV}, {256, matmulCoopQ41_256SPIRV}, {wideColumns, matmulCoopQ41_512SPIRV},
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
