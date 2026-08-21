// Records what ggml computes, so that the Go kernels can be checked against it
// without llama.cpp being present at test time.
//
// Three cases are written: a Q4_0 matrix-vector product performed by ggml
// itself (which quantizes the activation to Q8_0 internally, exactly as the Go
// kernel will), and one dequantized slab each of Q4_0 and Q6_K.

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void write_file(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
    fwrite(data, 1, bytes, f);
    fclose(f);
}

// A deterministic activation: the same values every run, on any machine.
static std::vector<float> activation(int64_t n) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(n);
    for (int64_t i = 0; i < n; ++i) x[i] = dist(rng);
    return x;
}

// Copies the first `rows` rows of a tensor, keeping its quantized bytes intact.
static std::vector<uint8_t> take_rows(const ggml_tensor * t, int64_t rows) {
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    std::vector<uint8_t> out(row_bytes * rows);
    memcpy(out.data(), t->data, out.size());
    return out;
}

static void dequantize(const ggml_tensor * t, int64_t rows, std::vector<float> & out) {
    const auto * traits = ggml_get_type_traits(t->type);
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    out.resize(rows * t->ne[0]);
    for (int64_t r = 0; r < rows; ++r) {
        traits->to_float((const uint8_t *) t->data + r * row_bytes,
                         out.data() + r * t->ne[0], t->ne[0]);
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: dump_quants <model.gguf> <output-dir>\n");
        return 1;
    }
    const std::string model = argv[1];
    const std::string dir   = std::string(argv[2]) + "/";

    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc=*/ false, /*ctx=*/ &meta };
    gguf_context * gguf = gguf_init_from_file(model.c_str(), gp);
    if (!gguf) { fprintf(stderr, "cannot open %s\n", model.c_str()); return 1; }

    ggml_tensor * q = ggml_get_tensor(meta, "blk.0.attn_q.weight");
    ggml_tensor * e = ggml_get_tensor(meta, "token_embd.weight");
    if (!q || !e) { fprintf(stderr, "expected tensors are missing\n"); return 1; }

    const int64_t cols = q->ne[0];          // shared dimension, 1536 for E2B
    const int64_t rows = 64;                // enough to exercise the row loop

    // --- case 1: the matrix-vector product, as ggml performs it -------------
    {
        std::vector<uint8_t> w = take_rows(q, rows);
        std::vector<float>   x = activation(cols);

        const size_t bufsize = w.size() + x.size() * 4 + rows * 4
                             + ggml_tensor_overhead() * 8 + ggml_graph_overhead() + (1u << 20);
        ggml_init_params ip = { bufsize, nullptr, false };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * a = ggml_new_tensor_2d(ctx, q->type, cols, rows);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, 1);
        memcpy(a->data, w.data(), w.size());
        memcpy(b->data, x.data(), x.size() * 4);

        ggml_tensor * y  = ggml_mul_mat(ctx, a, b);
        ggml_cgraph  * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        write_file(dir + "q4_0_matvec.w.bin", w.data(), w.size());
        write_file(dir + "q4_0_matvec.x.bin", x.data(), x.size() * 4);
        write_file(dir + "q4_0_matvec.y.bin", y->data, rows * 4);
        ggml_free(ctx);
    }

    // --- case 2 and 3: dequantization of a few rows -------------------------
    {
        std::vector<uint8_t> w = take_rows(q, 4);
        std::vector<float>   y; dequantize(q, 4, y);
        write_file(dir + "q4_0_dequant.w.bin", w.data(), w.size());
        write_file(dir + "q4_0_dequant.y.bin", y.data(), y.size() * 4);
    }
    {
        std::vector<uint8_t> w = take_rows(e, 4);
        std::vector<float>   y; dequantize(e, 4, y);
        write_file(dir + "q6_k_dequant.w.bin", w.data(), w.size());
        write_file(dir + "q6_k_dequant.y.bin", y.data(), y.size() * 4);
    }

    FILE * idx = fopen((dir + "index.json").c_str(), "w");
    fprintf(idx,
        "{\n"
        "  \"q4_0_matvec\":  {\"tensor\": \"blk.0.attn_q.weight\", \"type\": \"Q4_0\", \"rows\": %lld, \"cols\": %lld,\n"
        "                     \"weights\": \"q4_0_matvec.w.bin\", \"x\": \"q4_0_matvec.x.bin\", \"y\": \"q4_0_matvec.y.bin\"},\n"
        "  \"q4_0_dequant\": {\"tensor\": \"blk.0.attn_q.weight\", \"type\": \"Q4_0\", \"rows\": 4, \"cols\": %lld,\n"
        "                     \"weights\": \"q4_0_dequant.w.bin\", \"y\": \"q4_0_dequant.y.bin\"},\n"
        "  \"q6_k_dequant\": {\"tensor\": \"token_embd.weight\", \"type\": \"Q6_K\", \"rows\": 4, \"cols\": %lld,\n"
        "                     \"weights\": \"q6_k_dequant.w.bin\", \"y\": \"q6_k_dequant.y.bin\"}\n"
        "}\n",
        (long long) rows, (long long) cols, (long long) cols, (long long) e->ne[0]);
    fclose(idx);

    gguf_free(gguf);
    ggml_free(meta);
    printf("fixtures written to %s\n", dir.c_str());
    return 0;
}
