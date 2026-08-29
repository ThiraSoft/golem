// Records what ggml's antialiased bilinear resize makes of a grid, so the Go
// transcription can be checked against ggml rather than against a second
// reading of the same loop.
//
// It takes no model. clip.cpp resizes a learned position table with this when
// the patch grid is not the one the table was trained at, which for a tower
// with dynamic resolution is almost always. The antialias is the part worth
// pinning: the filter's support widens as the picture shrinks, and the weights
// are normalized by what was actually gathered rather than by their nominal
// sum. Neither shows on a picture and both move an embedding.
//
// Usage: dump_interpolate <out_dir>

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

// A deterministic grid: the same values every run, on any machine.
static std::vector<float> grid(int64_t n) {
    std::mt19937 rng(4321);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(n);
    for (int64_t i = 0; i < n; ++i) x[i] = dist(rng);
    return x;
}

struct Case {
    const char * name;
    int dst_w, dst_h;
};

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: dump_interpolate <out_dir>\n"); return 1; }
    const std::string out = argv[1];

    // The table this stands in for is 48x48 of 1152 channels. Three channels
    // here: the kernel is per channel, and a small count keeps the fixture in
    // kilobytes without changing what is being measured.
    const int src_w = 48, src_h = 48, channels = 3;

    const std::vector<Case> cases = {
        { "same",       48, 48 },  // the identity clip.cpp short-circuits
        { "down_small", 32, 32 },  // support widens, the antialias branch matters
        { "down_wide",  64, 16 },  // the two axes scale opposite ways
        { "up",         64, 64 },  // support clamps at 1, ordinary bilinear
        { "odd",        37, 23 },  // non-integer ratios, where a half-pixel shows
    };

    std::string index = "{\n  \"src_w\": " + std::to_string(src_w) +
                        ",\n  \"src_h\": " + std::to_string(src_h) +
                        ",\n  \"channels\": " + std::to_string(channels) +
                        ",\n  \"cases\": [\n";

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case & c = cases[ci];

        struct ggml_init_params ip = {
            /*.mem_size   =*/ 256u*1024u*1024u,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context * ctx = ggml_init(ip);

        // [w, h, c, 1], which is the layout resize_position_embeddings hands
        // ggml_interpolate after its permute.
        struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, src_w, src_h, channels, 1);
        std::vector<float> x = grid(ggml_nelements(a));
        memcpy(a->data, x.data(), x.size()*sizeof(float));

        struct ggml_tensor * r = ggml_interpolate(
            ctx, a, c.dst_w, c.dst_h, channels, 1,
            GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        write_file(out + "/" + c.name + ".x.bin", x.data(), x.size()*sizeof(float));
        write_file(out + "/" + c.name + ".y.bin", r->data, ggml_nbytes(r));

        index += std::string("    {\"name\": \"") + c.name +
                 "\", \"dst_w\": " + std::to_string(c.dst_w) +
                 ", \"dst_h\": " + std::to_string(c.dst_h) + "}";
        index += (ci + 1 == cases.size()) ? "\n" : ",\n";

        ggml_free(ctx);
        printf("%s: %dx%d -> %dx%d\n", c.name, src_w, src_h, c.dst_w, c.dst_h);
    }

    index += "  ]\n}\n";
    write_file(out + "/index.json", index.data(), index.size());
    return 0;
}
