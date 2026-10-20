#include "backend.h"
#include "dump_common.h"
#include "vae.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    // Force Vulkan0 backend device as required by the reference harness
    if (!backend_set_device("Vulkan0")) {
        fprintf(stderr, "FATAL: Vulkan0 backend device not found\n");
        return 1;
    }

    std::string model_path = "/mnt/data/LLMs_models/Serveurperso/YuE2-GGUF/YuE2-Vae-F32.gguf";
    std::string latent_path = "/mnt/data/golem-testdata/yue2/bench/q8-replay.vae";
    std::string out_dir = "/mnt/data/golem-testdata/yue2";

    const char * env_dir = getenv("GOLEM_YUE2_DIR");
    if (env_dir && strlen(env_dir) > 0) {
        model_path = std::string(env_dir) + "/YuE2-Vae-F32.gguf";
    }

    if (argc > 1) model_path = argv[1];
    if (argc > 2) latent_path = argv[2];
    if (argc > 3) out_dir = argv[3];

    fprintf(stderr, "[Dump-VAE] Model:  %s\n", model_path.c_str());
    fprintf(stderr, "[Dump-VAE] Latent: %s\n", latent_path.c_str());
    fprintf(stderr, "[Dump-VAE] OutDir: %s\n", out_dir.c_str());

    // Read full latent file
    FILE * f = fopen(latent_path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "FATAL: cannot open latent file %s\n", latent_path.c_str());
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t total_floats = (size_t) fsize / sizeof(float);
    std::vector<float> full_latent(total_floats);
    if (fread(full_latent.data(), sizeof(float), total_floats, f) != total_floats) {
        fprintf(stderr, "FATAL: cannot read %zu floats from %s\n", total_floats, latent_path.c_str());
        fclose(f);
        return 1;
    }
    fclose(f);

    size_t total_frames = total_floats / 64;
    fprintf(stderr, "[Dump-VAE] Loaded %zu frames (%zu floats)\n", total_frames, total_floats);
    if (total_frames < 2600) {
        fprintf(stderr, "FATAL: latent file has %zu frames, need at least 2600\n", total_frames);
        return 1;
    }

    Fixtures fixtures(out_dir);

    // Load VAE model
    VAEGGML vae = {};
    vae_ggml_load(&vae, model_path.c_str());

    // -------------------------------------------------------------
    // Stage 1: Untiled decode of frames 1000..1063 (T = 64)
    // -------------------------------------------------------------
    int T_short = 64;
    std::vector<float> latent_64((size_t) T_short * 64);
    memcpy(latent_64.data(), full_latent.data() + 1000 * 64, latent_64.size() * sizeof(float));
    fixtures.write("vae/latent", latent_64.data(), {T_short, 64});

    // Build graph manually to mark waypoints as outputs
    size_t ctx_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false);
    uint8_t * graph_buf = (uint8_t *) malloc(ctx_size);
    struct ggml_init_params p = { ctx_size, graph_buf, true };
    struct ggml_context * ctx = ggml_init(p);

    struct ggml_tensor * graph_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_short, 64);
    ggml_set_name(graph_input, "vae_input");
    ggml_set_input(graph_input);

    // conv1: [T, 64] -> [T, 2048]
    struct ggml_tensor * conv1_out = vae_conv1d(ctx, vae.c1w, vae.c1b, graph_input, 1, 3, 1);
    ggml_set_name(conv1_out, "vae_conv1");
    ggml_set_output(conv1_out);

    struct ggml_tensor * x = conv1_out;
    struct ggml_tensor * block_outs[6];

    for (int i = 0; i < 6; i++) {
        VAEBlock & b = vae.blk[i];
        x = vae_snake(ctx, x, b.sa, b.sb);
        int pad = (b.stride + 1) / 2;
        x = vae_conv_t1d(ctx, b.ctw, b.ctb, x, b.stride, pad, b.out_ch);
        for (int r = 0; r < 3; r++) {
            x = vae_res_unit(ctx, &b.ru[r], x);
        }
        std::string bname = "vae_block_" + std::to_string(i);
        ggml_set_name(x, bname.c_str());
        ggml_set_output(x);
        block_outs[i] = x;
    }

    x = vae_snake(ctx, x, vae.sa, vae.sb);
    struct ggml_tensor * audio_out = vae_conv1d(ctx, vae.c2w, NULL, x, 1, 3, 1);
    ggml_set_name(audio_out, "vae_audio");
    ggml_set_output(audio_out);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    for (int i = 0; i < 6; i++) {
        ggml_build_forward_expand(graph, block_outs[i]);
    }
    ggml_build_forward_expand(graph, conv1_out);
    ggml_build_forward_expand(graph, audio_out);

    if (!ggml_backend_sched_alloc_graph(vae.sched, graph)) {
        fprintf(stderr, "FATAL: graph alloc failed for T=64\n");
        return 1;
    }

    // Set input transposed to channel-major [T, 64]
    std::vector<float> scratch_in((size_t) T_short * 64);
    for (int c = 0; c < 64; c++) {
        for (int t = 0; t < T_short; t++) {
            scratch_in[c * T_short + t] = latent_64[t * 64 + c];
        }
    }
    ggml_backend_tensor_set(graph_input, scratch_in.data(), 0, scratch_in.size() * sizeof(float));

    // Compute graph
    ggml_backend_sched_graph_compute(vae.sched, graph);

    // Read back conv1
    int conv1_oc = (int) conv1_out->ne[1]; // 2048
    int conv1_ol = (int) conv1_out->ne[0]; // 64
    std::vector<float> conv1_data((size_t) conv1_oc * conv1_ol);
    ggml_backend_tensor_get(conv1_out, conv1_data.data(), 0, conv1_data.size() * sizeof(float));
    fixtures.write("vae/conv1", conv1_data.data(), {conv1_oc, conv1_ol});

    // Read back blocks 0..5
    for (int i = 0; i < 6; i++) {
        int oc = (int) block_outs[i]->ne[1];
        int ol = (int) block_outs[i]->ne[0];
        std::vector<float> blk_data((size_t) oc * ol);
        ggml_backend_tensor_get(block_outs[i], blk_data.data(), 0, blk_data.size() * sizeof(float));
        fixtures.write("vae/block_" + std::to_string(i), blk_data.data(), {oc, ol});
    }

    // Read back audio
    int audio_oc = (int) audio_out->ne[1]; // 2
    int audio_ol = (int) audio_out->ne[0]; // 122816
    std::vector<float> audio_data((size_t) audio_oc * audio_ol);
    ggml_backend_tensor_get(audio_out, audio_data.data(), 0, audio_data.size() * sizeof(float));
    fixtures.write("vae/audio", audio_data.data(), {audio_oc, audio_ol});

    // Clean up short graph
    ggml_backend_sched_reset(vae.sched);
    ggml_free(ctx);
    free(graph_buf);

    // -------------------------------------------------------------
    // Stage 2: Tiled decode of frames 2000..2599 (T = 600)
    // -------------------------------------------------------------
    int T_tiled = 600;
    std::vector<float> latent_600((size_t) T_tiled * 64);
    memcpy(latent_600.data(), full_latent.data() + 2000 * 64, latent_600.size() * sizeof(float));
    fixtures.write("vae_tiled/latent", latent_600.data(), {T_tiled, 64});

    int max_T_audio = T_tiled * 1920;
    std::vector<float> tiled_audio((size_t) 2 * max_T_audio);
    int T_audio_tiled = vae_ggml_decode_tiled(&vae, latent_600.data(), T_tiled, tiled_audio.data(), max_T_audio, 512, 16);
    if (T_audio_tiled < 0) {
        fprintf(stderr, "FATAL: vae_ggml_decode_tiled failed\n");
        return 1;
    }
    int want_T_audio_tiled = T_tiled * 1920 - 64;
    if (T_audio_tiled != want_T_audio_tiled) {
        fprintf(stderr, "FATAL: vae_ggml_decode_tiled returned %d, want %d\n", T_audio_tiled, want_T_audio_tiled);
        return 1;
    }
    fixtures.write("vae_tiled/audio", tiled_audio.data(), {2, T_audio_tiled});

    vae_ggml_free(&vae);
    fprintf(stderr, "[Dump-VAE] Done successfully.\n");
    return 0;
}
