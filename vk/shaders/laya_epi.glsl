// What shaders/laya_mm.comp and shaders/laya_reduce.comp do with a finished
// sum. The product writes into Y, a float32 buffer, or into HL, an operand
// split for the next product: its high part, then at p.olo what is left.
//
//   0  store:  Y = sum + bias
//   1  add:    Y += sum + bias, the residual stream taking a block's output
//   2  geglu:  the rows come in pairs, an input and its gate (the host
//              interleaves ModernBERT's two halves when it uploads them), and
//              GELU(input) * gate goes to HL, a row half as wide
//   3  relu:   max(sum + bias, 0) to HL
//   4  gelu:   Y = GELU(sum + bias), the scorer's hidden layer, which the
//              host reads back

#define EPI_STORE 0u
#define EPI_ADD   1u
#define EPI_GEGLU 2u
#define EPI_RELU  3u
#define EPI_GELU  4u

layout(push_constant) uniform Push {
    uint outputs;
    uint inputs;
    uint cols;
    uint base;    // where W starts, in halves
    uint bias;    // where the bias starts in par, or ~0
    uint mode;
    uint lo;      // where the operand's low plane starts in X, in halves
    uint slices;  // 0 or 1 for a product that is not split
    uint kper;    // the shared dimension one slice walks, a whole number of BK
    uint stride;  // floats between two slices' partial sums
    uint olo;     // where the written low plane starts in HL, in halves
} p;

// GLSL has no erf, so this is Numerical Recipes' erfc, whose relative error
// is under 1.2e-7 everywhere: the GELU PyTorch computes by default is the
// one on the error function.
float erfc_(float x) {
    float z = abs(x);
    float t = 1.0 / (1.0 + 0.5 * z);
    float r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
        t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0.0 ? r : 2.0 - r;
}

float gelu(float x) { return 0.5 * x * erfc_(-x * 0.70710678118654752); }

// split writes v as the next product reads it. The high part is v cut to
// fp16's ten bits of mantissa by a mask rather than by a rounding: the driver
// folds a conversion to fp16 and back into nothing, which would leave the
// remainder zero.
void split(uint at, float v) {
    float top = uintBitsToFloat(floatBitsToUint(v) & 0xffffe000u);
    hl[at] = float16_t(top);
    hl[p.olo + at] = float16_t(v - top);
}

float biasOf(uint row) { return (p.bias == ~0u) ? 0.0 : par[p.bias + row]; }

void epiOne(uint row, uint col, float v) {
    v += biasOf(row);
    uint at = col * p.outputs + row;
    if (p.mode == EPI_STORE) {
        y[at] = v;
    } else if (p.mode == EPI_ADD) {
        y[at] += v;
    } else if (p.mode == EPI_GELU) {
        y[at] = gelu(v);
    } else {
        split(at, max(v, 0.0));
    }
}

void epiPair(uint row, uint col, float a, float g) {
    a += biasOf(row);
    g += biasOf(row + 1u);
    split(col * (p.outputs / 2u) + row / 2u, gelu(a) * g);
}
