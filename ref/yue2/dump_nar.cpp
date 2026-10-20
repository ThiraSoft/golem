#include "backend.h"
#include "bpe.h"
#include "dump_common.h"
#include "nar.h"
#include "prompt.h"
#include "qwen3-lm.h"
#include "request.h"
#include "torch-cpu-rng.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// parse_tokens extracts comma-separated integer token values from a CSV string.
static bool parse_tokens(const std::string & csv, std::vector<int> * out) {
    out->clear();
    const char * p = csv.c_str();
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
            p++;
        }
        if (!*p) {
            break;
        }
        char * end   = nullptr;
        long   value = strtol(p, &end, 10);
        if (end == p || value < 0 || value >= YUE2_CODEC_SIZE) {
            fprintf(stderr, "[Dump-NAR] FATAL: semantic token outside [0, %d)\n", YUE2_CODEC_SIZE);
            return false;
        }
        out->push_back((int) value);
        p = end;
    }
    return !out->empty();
}


// read_raw reads a whole fixture file written by an earlier run.
static bool read_raw(const std::string & path, void * dst, size_t bytes) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "FATAL: cannot open %s\n", path.c_str());
        return false;
    }
    size_t got = fread(dst, 1, bytes, f);
    bool   ok  = got == bytes && fgetc(f) == EOF;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "FATAL: %s is not %zu bytes\n", path.c_str(), bytes);
    }
    return ok;
}

// record_exact replays the first velocity evaluation of the default run with
// the products summing in float32 over fp16 operands, and writes its probes
// under nar_exact/.
//
// At its default settings ggml-vulkan quantizes the activations of every
// product against Q8_0 weights to Q8_1 (an fp16 scale and an int8 a value,
// blocks of 32), because this card has integer dot products; that alone puts
// the default run's hidden_after_input 0.34% away from the exact sum of its
// fp16 noise and Q8_0 weights. The process is started with integer dot
// products and cooperative matrices disabled (the int8 matrix cores are the
// other route to Q8_1), and every MUL_MAT is set to GGML_PREC_F32, so the
// products take the weights dequantized to fp16 and the activations rounded
// to fp16, and sum them in float32.
//
// The AR cache is not prefilled again under these settings: the default
// run's kv_all_{k,v}.f16 are loaded into it, so that both recordings share
// the prefix golem is given.
static int record_exact(const std::string & model_path, const std::string & out_dir) {
    Yue2NAR nar;
    if (!nar_load(&nar, model_path.c_str(), {})) {
        fprintf(stderr, "FATAL: failed to load NAR from %s\n", model_path.c_str());
        return 1;
    }

    // The default run's prefix length, noise and cache rows.
    std::string dir = out_dir + "/nar/";
    FILE *      f   = fopen((dir + "ar_ids").c_str(), "rb");
    if (!f) {
        fprintf(stderr, "FATAL: no %sar_ids: run the default recording first\n", dir.c_str());
        return 1;
    }
    fseek(f, 0, SEEK_END);
    int ar_len = (int) (ftell(f) / 4);
    fclose(f);

    int T_lat = 100;
    int M     = 1;
    std::vector<float> state((size_t) nar.latent_dim * T_lat * M);
    if (!read_raw(dir + "noise", state.data(), state.size() * sizeof(float))) {
        return 1;
    }

    Qw3lmKvCache  kv;
    Qwen3LMConfig kv_cfg = nar.cfg;
    kv_cfg.max_seq_len   = 4096;
    qw3lm_kv_init(&kv, kv_cfg, nar.backend);
    if (!qw3lm_kv_sets(&kv, 1)) {
        fprintf(stderr, "FATAL: failed to allocate KV cache sets\n");
        return 1;
    }
    int D   = kv_cfg.head_dim;
    int S   = kv_cfg.max_seq_len;
    int Nkv = kv_cfg.n_kv_heads;
    int L   = kv_cfg.n_layers;
    std::vector<uint16_t> all_k((size_t) L * ar_len * Nkv * D), all_v(all_k.size());
    if (!read_raw(dir + "kv_all_k.f16", all_k.data(), all_k.size() * 2) ||
        !read_raw(dir + "kv_all_v.f16", all_v.data(), all_v.size() * 2)) {
        return 1;
    }
    std::vector<uint16_t> k_raw((size_t) D * S * Nkv, 0), v_raw(k_raw.size(), 0);
    for (int l = 0; l < L; l++) {
        for (int pos = 0; pos < ar_len; pos++) {
            for (int head = 0; head < Nkv; head++) {
                for (int d = 0; d < D; d++) {
                    size_t dst = (size_t) head * S * D + (size_t) pos * D + d;
                    size_t src = (size_t) l * ar_len * Nkv * D + (size_t) pos * Nkv * D + (size_t) head * D + d;
                    k_raw[dst] = all_k[src];
                    v_raw[dst] = all_v[src];
                }
            }
        }
        ggml_backend_tensor_set(kv.k4[l], k_raw.data(), 0, k_raw.size() * 2);
        ggml_backend_tensor_set(kv.v4[l], v_raw.data(), 0, v_raw.size() * 2);
    }
    kv.pos[0] = ar_len;

    // nar_velocity, with the products set to float32 between the graph's
    // build and its allocation.
    float raw_t = nar_logit_clamped(1.0f);
    if (!nar_build_graph(&nar, &kv, T_lat, M, ar_len, 0)) {
        return 1;
    }
    int f32 = 0;
    for (int i = 0; i < ggml_graph_n_nodes(nar.graph); i++) {
        struct ggml_tensor * n = ggml_graph_node(nar.graph, i);
        if (n->op == GGML_OP_MUL_MAT) {
            ggml_mul_mat_set_prec(n, GGML_PREC_F32);
            f32++;
        }
    }
    fprintf(stderr, "[Dump-NAR] %d products at GGML_PREC_F32\n", f32);
    ggml_backend_sched_reset(nar.sched);
    if (!ggml_backend_sched_alloc_graph(nar.sched, nar.graph)) {
        fprintf(stderr, "FATAL: graph alloc failed\n");
        return 1;
    }
    int                N = T_lat + 2;
    std::vector<float> x((size_t) nar.latent_dim * N, 0.0f);
    memcpy(x.data() + nar.latent_dim, state.data(), state.size() * sizeof(float));
    ggml_backend_tensor_set(nar.in_x, x.data(), 0, x.size() * sizeof(float));
    float feats[YUE2_TIME_EMBED_DIM];
    nar_time_features(nar_shift_t(&nar, raw_t), feats);
    ggml_backend_tensor_set(nar.in_time, feats, 0, sizeof(feats));
    ggml_backend_sched_graph_compute(nar.sched, nar.graph);
    std::vector<float> first(state.size());
    ggml_backend_tensor_get(nar.out_v, first.data(), 0, first.size() * sizeof(float));

    Fixtures fixtures(out_dir);
    const int                probes[] = YUE2_NAR_PROBE_LAYERS;
    const char *             fixed[]  = { "temb_t", "hidden_after_input", "layer0_sa_output" };
    std::vector<std::string> names(fixed, fixed + 3);
    for (int probe : probes) {
        names.push_back("hidden_after_layer" + std::to_string(probe));
    }
    for (const std::string & name : names) {
        struct ggml_tensor * t = ggml_graph_get_tensor(nar.graph, name.c_str());
        if (!t) {
            fprintf(stderr, "FATAL: probe tensor %s not found in graph\n", name.c_str());
            return 1;
        }
        int64_t            n0 = t->ne[0];
        int64_t            n1 = t->ne[1];
        std::vector<float> buf((size_t) n0 * n1);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        if (n1 <= 1) {
            fixtures.write("nar_exact/" + name, buf.data(), { (int) n0 });
        } else {
            fixtures.write("nar_exact/" + name, buf.data(), { (int) n1, (int) n0 });
        }
    }
    fixtures.write("nar_exact/v_first_0", first.data(), { T_lat, nar.latent_dim });

    nar_free(&nar);
    qw3lm_kv_free(&kv);
    fprintf(stderr, "[Dump-NAR] Exact recording done.\n");
    return 0;
}

int main(int argc, char ** argv) {
    // "exact" as the fourth argument records nar_exact/ (see record_exact);
    // the switches must be set before ggml-vulkan looks at the device.
    bool exact = argc > 4 && strcmp(argv[4], "exact") == 0;
    if (exact) {
        setenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1", 1);
        setenv("GGML_VK_DISABLE_COOPMAT", "1", 1);
    }

    // Force Vulkan0 backend device as required by the reference harness
    if (!backend_set_device("Vulkan0")) {
        fprintf(stderr, "FATAL: Vulkan0 backend device not found\n");
        return 1;
    }

    std::string model_path   = "/mnt/data/LLMs_models/Serveurperso/YuE2-GGUF/YuE2-3B-Q8_0.gguf";
    std::string request_path = "ref/yue2/requests/nar-short.json";
    std::string out_dir      = "/mnt/data/golem-testdata/yue2";

    const char * env_dir = getenv("GOLEM_YUE2_DIR");
    if (env_dir && strlen(env_dir) > 0) {
        model_path = std::string(env_dir) + "/YuE2-3B-Q8_0.gguf";
    }

    if (argc > 1) model_path   = argv[1];
    if (argc > 2) request_path = argv[2];
    if (argc > 3) out_dir      = argv[3];

    fprintf(stderr, "[Dump-NAR] Model:   %s\n", model_path.c_str());
    fprintf(stderr, "[Dump-NAR] Request: %s\n", request_path.c_str());
    fprintf(stderr, "[Dump-NAR] OutDir:  %s\n", out_dir.c_str());

    if (exact) {
        return record_exact(model_path, out_dir);
    }

    // Parse the JSON request
    Yue2Request req;
    request_init(&req);
    if (!request_parse(&req, request_path.c_str())) {
        fprintf(stderr, "FATAL: failed to parse request %s\n", request_path.c_str());
        return 1;
    }

    // Load BPE tokenizer from the GGUF backbone
    BPETokenizer tok;
    if (!load_bpe_from_gguf(&tok, model_path.c_str())) {
        fprintf(stderr, "FATAL: failed to load BPE tokenizer from %s\n", model_path.c_str());
        return 1;
    }
    auto encode = [&tok](const std::string & text) {
        return bpe_encode(&tok, text);
    };

    // Encode ABC score and construct full conditional prompt prefix
    std::vector<int> abc_ids = encode(req.abc);
    std::vector<int> prefix  = yue2_build_prompt_ids(encode, YUE2_COT_FULL, req.style, req.lyrics, &abc_ids);

    // Parse the semantic tokens (first 100 codes)
    std::vector<int> values;
    if (!parse_tokens(req.semantic_tokens, &values)) {
        fprintf(stderr, "FATAL: failed to parse semantic tokens\n");
        return 1;
    }
    if (values.size() != 100) {
        fprintf(stderr, "FATAL: expected 100 semantic tokens, got %zu\n", values.size());
        return 1;
    }

    // Full AR sequence: prefix, 100 codes with codec offset, then music end
    std::vector<int> sequence = prefix;
    for (int v : values) {
        sequence.push_back(v + YUE2_CODEC_OFFSET);
    }
    sequence.push_back(YUE2_MUSIC_END);
    int ar_len = (int) sequence.size();
    fprintf(stderr, "[Dump-NAR] AR prefix len: %d (prefix %zu, codes %zu, end 1)\n",
            ar_len, prefix.size(), values.size());

    Fixtures fixtures(out_dir);

    // 1. Write nar/ar_ids
    fixtures.write_i32("nar/ar_ids", sequence.data(), {ar_len});

    // 2. Write standalone RNG check fixtures
    std::vector<float> rng_37(37);
    torch_cpu_randn((uint64_t) req.seed, rng_37.data(), 37);
    fixtures.write("rng/normal_37", rng_37.data(), {37});

    std::vector<float> rng_6400(6400);
    torch_cpu_randn((uint64_t) req.seed, rng_6400.data(), 6400);
    fixtures.write("rng/normal_6400", rng_6400.data(), {6400});

    // Load AR model for prefill
    Qwen3LM lm;
    if (!qw3lm_load(&lm, model_path.c_str(), {})) {
        fprintf(stderr, "FATAL: failed to load LM from %s\n", model_path.c_str());
        return 1;
    }

    // Allocate KV cache with max_seq = 4096 to keep memory small
    Qw3lmKvCache  kv;
    Qwen3LMConfig kv_cfg = lm.cfg;
    kv_cfg.max_seq_len   = 4096;
    qw3lm_kv_init(&kv, kv_cfg, lm.backend);
    if (!qw3lm_kv_sets(&kv, 1)) {
        fprintf(stderr, "FATAL: failed to allocate KV cache sets\n");
        return 1;
    }

    // Prefill the AR sequence into KV cache set 0
    std::vector<float> logits(lm.cfg.vocab_size);
    qw3lm_forward(&lm, &kv, sequence.data(), ar_len, 0, logits.data(), 0, lm.cfg.vocab_size);
    fprintf(stderr, "[Dump-NAR] Prefilled %d tokens into KV cache (pos = %d)\n", ar_len, kv.pos[0]);

    // Dump KV cache rows
    int D   = kv.cfg.head_dim;    // 128
    int S   = kv.cfg.max_seq_len; // 4096
    int Nkv = kv.cfg.n_kv_heads;  // 8
    int L   = kv.cfg.n_layers;    // 28

    std::vector<uint16_t>    kv_all_k((size_t) L * ar_len * Nkv * D);
    std::vector<uint16_t>    kv_all_v((size_t) L * ar_len * Nkv * D);
    std::vector<ggml_fp16_t> k_raw((size_t) D * S * Nkv);
    std::vector<ggml_fp16_t> v_raw((size_t) D * S * Nkv);

    for (int l = 0; l < L; l++) {
        ggml_backend_tensor_get(kv.k4[l], k_raw.data(), 0, k_raw.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(kv.v4[l], v_raw.data(), 0, v_raw.size() * sizeof(ggml_fp16_t));

        for (int pos = 0; pos < ar_len; pos++) {
            for (int head = 0; head < Nkv; head++) {
                for (int d = 0; d < D; d++) {
                    size_t src_idx = (size_t) head * S * D + (size_t) pos * D + d;
                    size_t dst_idx = (size_t) l * ar_len * Nkv * D + (size_t) pos * Nkv * D + (size_t) head * D + d;
                    kv_all_k[dst_idx] = (uint16_t) k_raw[src_idx];
                    kv_all_v[dst_idx] = (uint16_t) v_raw[src_idx];
                }
            }
        }

        if (l == 0 || l == 13 || l == 27) {
            std::vector<float> k_f32((size_t) ar_len * Nkv * D);
            std::vector<float> v_f32((size_t) ar_len * Nkv * D);
            for (int pos = 0; pos < ar_len; pos++) {
                for (int head = 0; head < Nkv; head++) {
                    for (int d = 0; d < D; d++) {
                        size_t src_idx = (size_t) head * S * D + (size_t) pos * D + d;
                        size_t dst_idx = (size_t) pos * Nkv * D + (size_t) head * D + d;
                        k_f32[dst_idx] = ggml_fp16_to_fp32(k_raw[src_idx]);
                        v_f32[dst_idx] = ggml_fp16_to_fp32(v_raw[src_idx]);
                    }
                }
            }
            fixtures.write("nar/kv_k_l" + std::to_string(l), k_f32.data(), {ar_len, Nkv, D});
            fixtures.write("nar/kv_v_l" + std::to_string(l), v_f32.data(), {ar_len, Nkv, D});
        }
    }

    fixtures.write_f16("nar/kv_all_k.f16", kv_all_k.data(), {L, ar_len, Nkv, D});
    fixtures.write_f16("nar/kv_all_v.f16", kv_all_v.data(), {L, ar_len, Nkv, D});

    // Load NAR model
    Yue2NAR nar;
    if (!nar_load(&nar, model_path.c_str(), {})) {
        fprintf(stderr, "FATAL: failed to load NAR from %s\n", model_path.c_str());
        qw3lm_kv_free(&kv);
        qw3lm_free(&lm);
        return 1;
    }

    // Free AR LM weights to save memory
    qw3lm_free(&lm);

    // Initial noise for 1 variation of 100 frames
    int T_lat = 100;
    int M     = 1;
    std::vector<float> state((size_t) nar.latent_dim * T_lat * M);
    torch_cpu_randn((uint64_t) req.seed, state.data(), (int64_t) state.size());
    fixtures.write("nar/noise", state.data(), {T_lat, nar.latent_dim});

    // 32 midpoint solver steps
    int   steps = req.steps;
    float dt    = 1.0f / (float) steps;
    std::vector<float> first(state.size()), mid(state.size()), second(state.size());

    for (int step = 0; step < steps; step++) {
        float t = 1.0f - (float) step * dt;
        if (!nar_velocity(&nar, &kv, state.data(), T_lat, M, ar_len, 0, nar_logit_clamped(t), first.data())) {
            fprintf(stderr, "FATAL: nar_velocity failed at step %d\n", step);
            return 1;
        }

        // Record probes from the first evaluation
        if (step == 0) {
            const int probes[] = YUE2_NAR_PROBE_LAYERS;
            const char * fixed[] = { "temb_t", "hidden_after_input", "layer0_sa_output" };
            std::vector<std::string> names(fixed, fixed + 3);
            for (int probe : probes) {
                names.push_back("hidden_after_layer" + std::to_string(probe));
            }
            for (const std::string & name : names) {
                struct ggml_tensor * tens = ggml_graph_get_tensor(nar.graph, name.c_str());
                if (!tens) {
                    fprintf(stderr, "FATAL: probe tensor %s not found in graph\n", name.c_str());
                    return 1;
                }
                int64_t n0 = tens->ne[0];
                int64_t n1 = tens->ne[1];
                std::vector<float> buf((size_t) n0 * n1);
                ggml_backend_tensor_get(tens, buf.data(), 0, (size_t) n0 * n1 * sizeof(float));
                if (n1 <= 1) {
                    fixtures.write("nar/" + name, buf.data(), {(int) n0});
                } else {
                    fixtures.write("nar/" + name, buf.data(), {(int) n1, (int) n0});
                }
            }
            fixtures.write("nar/v_first_0", first.data(), {T_lat, nar.latent_dim});
        }

        for (size_t i = 0; i < state.size(); i++) {
            mid[i] = state[i] - first[i] * (dt * 0.5f);
        }
        if (!nar_velocity(&nar, &kv, mid.data(), T_lat, M, ar_len, 0, nar_logit_clamped(t - dt * 0.5f), second.data())) {
            fprintf(stderr, "FATAL: nar_velocity second call failed at step %d\n", step);
            return 1;
        }

        if (step == 0) {
            fixtures.write("nar/v_second_0", second.data(), {T_lat, nar.latent_dim});
        }

        for (size_t i = 0; i < state.size(); i++) {
            state[i] -= second[i] * dt;
        }

        if (step == 0) {
            fixtures.write("nar/xt_0", state.data(), {T_lat, nar.latent_dim});
        } else if (step == 1) {
            fixtures.write("nar/xt_1", state.data(), {T_lat, nar.latent_dim});
        } else if (step == 7) {
            fixtures.write("nar/xt_7", state.data(), {T_lat, nar.latent_dim});
        } else if (step == 31) {
            fixtures.write("nar/xt_31", state.data(), {T_lat, nar.latent_dim});
        }

        fprintf(stderr, "[Dump-NAR] Step %d/%d completed\n", step + 1, steps);
    }

    fixtures.write("nar/latent", state.data(), {T_lat, nar.latent_dim});

    nar_free(&nar);
    qw3lm_kv_free(&kv);

    fprintf(stderr, "[Dump-NAR] Done successfully.\n");
    return 0;
}
