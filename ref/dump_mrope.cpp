// Records what ggml's interleaved M-RoPE makes of a vector, so the Go rotation
// can be checked against ggml rather than against a second transcription of
// the same rule.
//
// It takes no model. The rotation depends on the sections, the base, the head
// width and the position, and on nothing a checkpoint holds — so the cases are
// written here, and the sections Qwen3.8-27B declares are one of them.
//
// Usage: dump_mrope <mrope_out_dir> [vision_out_dir]
//
// The second directory takes the vision tower's rotation, which is the same op
// under GGML_ROPE_TYPE_VISION and a different rule: there the frequency starts
// again at every section, where the trunk's never resets. Given one argument
// only the trunk's cases are written, so the command in ref/README.md that
// predates the tower goes on working.

#include "ggml.h"
#include "ggml-cpu.h"

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

// A deterministic input: the same values every run, on any machine.
static std::vector<float> activation(int64_t n) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(n);
    for (int64_t i = 0; i < n; ++i) x[i] = dist(rng);
    return x;
}

struct Case {
    const char * name;
    int t, h, w, e;
};

// record writes one set of cases under one rotation type into one directory.
static void record(const std::string & out, int head_size, int n_dims, int n_head,
                   float base, int sections[GGML_MROPE_SECTIONS], int rope_type,
                   const std::vector<Case> & cases) {
    std::string index = "{\n  \"head_size\": " + std::to_string(head_size) +
                        ",\n  \"n_dims\": " + std::to_string(n_dims) +
                        ",\n  \"n_head\": " + std::to_string(n_head) +
                        ",\n  \"base\": " + std::to_string(base) +
                        ",\n  \"sections\": [" + std::to_string(sections[0]) + ", " +
                        std::to_string(sections[1]) + ", " + std::to_string(sections[2]) + ", " +
                        std::to_string(sections[3]) + "],\n  \"cases\": [\n";

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case & c = cases[ci];

        struct ggml_init_params ip = {
            /*.mem_size   =*/ 64u*1024u*1024u,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, n_head, 1);
        std::vector<float> x = activation(ggml_nelements(a));
        memcpy(a->data, x.data(), x.size()*sizeof(float));

        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        ((int32_t *) b->data)[0] = c.t;
        ((int32_t *) b->data)[1] = c.h;
        ((int32_t *) b->data)[2] = c.w;
        ((int32_t *) b->data)[3] = c.e;

        struct ggml_tensor * r = ggml_rope_multi(
            ctx, a, b, NULL,
            n_dims, sections, rope_type,
            /*n_ctx_orig =*/ 0,
            /*freq_base  =*/ base,
            /*freq_scale =*/ 1.0f,
            /*ext_factor =*/ 0.0f,
            /*attn_factor=*/ 1.0f,
            /*beta_fast  =*/ 32.0f,
            /*beta_slow  =*/ 1.0f);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        write_file(out + "/" + c.name + ".x.bin", x.data(), x.size()*sizeof(float));
        write_file(out + "/" + c.name + ".y.bin", r->data, ggml_nbytes(r));

        index += std::string("    {\"name\": \"") + c.name + "\", \"t\": " + std::to_string(c.t) +
                 ", \"h\": " + std::to_string(c.h) + ", \"w\": " + std::to_string(c.w) +
                 ", \"e\": " + std::to_string(c.e) + "}";
        index += (ci + 1 == cases.size()) ? "\n" : ",\n";

        ggml_free(ctx);
        printf("%s: t=%d h=%d w=%d\n", c.name, c.t, c.h, c.w);
    }

    index += "  ]\n}\n";
    write_file(out + "/index.json", index.data(), index.size());
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: dump_mrope <mrope_out_dir> [vision_out_dir]\n");
        return 1;
    }
    const std::string out = argv[1];

    // Qwen3.8-27B's trunk: 64 rotated dimensions of a 256-wide head, sections
    // summing to 32, base ten million. The frequency never resets.
    int sections[GGML_MROPE_SECTIONS] = { 11, 11, 10, 0 };

    // Degenerate cases pin the claim text depends on. The spread ones pin the
    // round robin. The boundary ones sit where 3*sections[i] cuts a section
    // off, which is the arithmetic a transcription gets wrong silently.
    const std::vector<Case> cases = {
        { "degenerate_0",   0,   0,   0, 0 },
        { "degenerate_37", 37,  37,  37, 0 },
        { "spread_small",   4,   1,   2, 0 },
        { "spread_grid",   12,   3,   7, 0 },
        { "boundary_low",   1,   0,   0, 0 },
        { "boundary_high", 40,  31,  29, 0 },
        { "image_row",      9,   0,  15, 0 },
        { "image_col",      9,  15,   0, 0 },
    };
    record(out, 256, 64, 2, 1e7f, sections, GGML_ROPE_TYPE_IMROPE, cases);

    if (argc == 3) {
        // The vision tower: a head of 72 of which 36 rotate, four equal
        // sections, base ten thousand. VISION reads only the first two — the
        // row and the column — and restarts the frequency at each.
        int v_sections[GGML_MROPE_SECTIONS] = { 18, 18, 18, 18 };
        const std::vector<Case> v_cases = {
            { "v_origin",   0,  0, 0, 0 },
            { "v_row",      3,  0, 0, 0 },
            { "v_col",      0,  5, 0, 0 },
            { "v_grid",     7, 11, 0, 0 },
            { "v_far",     47, 47, 0, 0 },
        };
        record(argv[2], 72, 36, 2, 10000.0f, v_sections, GGML_ROPE_TYPE_VISION, v_cases);
    }
    return 0;
}
