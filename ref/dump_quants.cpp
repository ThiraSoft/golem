// Records what ggml computes, so that the Go kernels can be checked against it
// without llama.cpp being present at test time.
//
// For every quantized format the given model carries, two cases are written: a
// matrix-vector product performed by ggml itself (which quantizes the
// activation to whatever that weight's dot wants, exactly as the Go kernel
// will) and one dequantized slab.
//
// Which tensors those are depends on the model, and a model that has none of a
// format simply records nothing for it: Gemma 4 gives Q4_0 and Q6_K, and
// Qwen3.8-27B gives Q4_1 as well, on its ffn_down.

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cctype>
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

    // What to record, and where to look for it. A model that has none of a
    // format is not an error: it records nothing under that name, and the Go
    // test for it skips the way it already does when the fixtures are absent.
    struct wanted { const char * label; const char * tensor; };
    const wanted wants[] = {
        {"q4_0", "blk.0.attn_q.weight"},
        {"q4_1", "blk.0.ffn_down.weight"},
        {"q6_k", "token_embd.weight"},
    };

    std::string entries;
    for (const wanted & want : wants) {
        ggml_tensor * t = ggml_get_tensor(meta, want.tensor);
        if (!t) continue;
        std::string type = ggml_type_name(t->type);   // ggml spells it in lower case
        for (char & c : type) c = toupper(c);
        // The label names the format, so a tensor that is not in it belongs to
        // another model's recording and is skipped rather than mislabelled.
        std::string expected = want.label;
        for (char & c : expected) c = toupper(c);
        if (type != expected) {
            fprintf(stderr, "%s is %s, not %s: skipped\n", want.tensor, type.c_str(), expected.c_str());
            continue;
        }

        const int64_t cols = t->ne[0];
        const int64_t rows = 64;            // enough to exercise the row loop
        const std::string label = want.label;

        // --- the matrix-vector product, as ggml performs it -----------------
        {
            std::vector<uint8_t> w = take_rows(t, rows);
            std::vector<float>   x = activation(cols);

            const size_t bufsize = w.size() + x.size() * 4 + rows * 4
                                 + ggml_tensor_overhead() * 8 + ggml_graph_overhead() + (1u << 20);
            ggml_init_params ip = { bufsize, nullptr, false };
            ggml_context * ctx = ggml_init(ip);

            ggml_tensor * a = ggml_new_tensor_2d(ctx, t->type, cols, rows);
            ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, 1);
            memcpy(a->data, w.data(), w.size());
            memcpy(b->data, x.data(), x.size() * 4);

            ggml_tensor * y  = ggml_mul_mat(ctx, a, b);
            ggml_cgraph  * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            ggml_graph_compute_with_ctx(ctx, gf, 1);

            write_file(dir + label + "_matvec.w.bin", w.data(), w.size());
            write_file(dir + label + "_matvec.x.bin", x.data(), x.size() * 4);
            write_file(dir + label + "_matvec.y.bin", y->data, rows * 4);
            ggml_free(ctx);
        }

        // --- dequantization of a few rows -----------------------------------
        {
            std::vector<uint8_t> w = take_rows(t, 4);
            std::vector<float>   y; dequantize(t, 4, y);
            write_file(dir + label + "_dequant.w.bin", w.data(), w.size());
            write_file(dir + label + "_dequant.y.bin", y.data(), y.size() * 4);
        }

        char buf[1024];
        snprintf(buf, sizeof buf,
            "%s  \"%s_matvec\":  {\"tensor\": \"%s\", \"type\": \"%s\", \"rows\": %lld, \"cols\": %lld,\n"
            "                     \"weights\": \"%s_matvec.w.bin\", \"x\": \"%s_matvec.x.bin\", \"y\": \"%s_matvec.y.bin\"},\n"
            "  \"%s_dequant\": {\"tensor\": \"%s\", \"type\": \"%s\", \"rows\": 4, \"cols\": %lld,\n"
            "                     \"weights\": \"%s_dequant.w.bin\", \"y\": \"%s_dequant.y.bin\"}",
            entries.empty() ? "" : ",\n", label.c_str(), want.tensor, expected.c_str(),
            (long long) rows, (long long) cols, label.c_str(), label.c_str(), label.c_str(),
            label.c_str(), want.tensor, expected.c_str(), (long long) cols,
            label.c_str(), label.c_str());
        entries += buf;
    }
    if (entries.empty()) { fprintf(stderr, "no tensor of a recorded format in %s\n", model.c_str()); return 1; }

    FILE * idx = fopen((dir + "index.json").c_str(), "w");
    fprintf(idx, "{\n%s\n}\n", entries.c_str());
    fclose(idx);

    gguf_free(gguf);
    ggml_free(meta);
    printf("fixtures written to %s\n", dir.c_str());
    return 0;
}
