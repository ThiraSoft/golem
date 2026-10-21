#include "backend.h"
#include "bpe.h"
#include "dump_common.h"
#include "prompt.h"
#include "qwen3-lm.h"
#include "request.h"
#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The AR half of yue2.cpp on the Guerreros prompt: a prefill whose hidden
// states are tapped after three layers, the logits of its last position in
// both phase windows, eight decode steps of a real semantic draw, and a guided
// run (cot off, guidance 1.01) of four steps with both branches' logits.

static const int kTapLayers[3] = { 0, 13, 27 };
static const int kTapRows      = 4;  // the last positions of the prefill

// The exact recording (fourth argument "exact"): the process starts with
// integer dot products and cooperative matrices off and every MUL_MAT of the
// probe graph at GGML_PREC_F32, so no product quantizes its activation to
// Q8_1 (see dump_nar.cpp's record_exact). Only the prefill probes are
// written, under ar_exact/.
static bool g_exact = false;

// Probes of one prefill: the hidden state after each tapped layer and after
// the final norm, over the last kTapRows positions, and the logits of the
// last position in both phase windows.
struct PrefillProbes {
    std::vector<float> hidden[3];
    std::vector<float> normed;
    std::vector<float> abc, semantic;
};

// probe_forward is qw3lm_forward (src/qwen3-lm.h) with outputs set on the
// tapped nodes: the same graph, the same inputs, the KV set advanced the same
// way. The two heads read the same final hidden state.
static void probe_forward(Qwen3LM * m, Qw3lmKvCache * kv, const int * token_ids, int n_tokens, int kv_set,
                          PrefillProbes * out) {
    if (m->batch_graph.graph.sched_allocated) {
        static_graph_release(&m->batch_graph.graph, m->sched);
        m->batch_graph.built = false;
    }

    const Qwen3LMConfig & c      = m->cfg;
    int                   H      = c.hidden_size;
    int                   kv_pos = kv->pos[kv_set];
    int                   kv_len = kv_pos + n_tokens;

    const int max_seq = kv->cfg.max_seq_len;
    if (kv_len > max_seq || n_tokens < kTapRows) {
        fprintf(stderr, "[Dump-AR] FATAL: kv_len %d > max_seq %d or %d tokens\n", kv_len, max_seq, n_tokens);
        exit(1);
    }
    const int kv_pad_raw = (int) GGML_PAD(kv_len, 256);
    const int n_kv_pad   = kv_pad_raw < max_seq ? kv_pad_raw : max_seq;

    struct ggml_context * ctx = graph_arena_begin(&m->arena_prefill);
    struct ggml_cgraph *  gf  = ggml_new_graph_custom(ctx, QW3LM_GRAPH_NODES, false);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, n_tokens);
    ggml_set_name(mask, "causal_mask");
    ggml_set_input(mask);
    struct ggml_tensor * token_ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token_ids_t, "token_ids");
    ggml_set_input(token_ids_t);
    struct ggml_tensor * hidden  = ggml_get_rows(ctx, m->embed_tokens, token_ids_t);
    struct ggml_tensor * kv_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_name(kv_rows, "kv_rows");
    ggml_set_input(kv_rows);

    struct ggml_tensor * taps[3] = { nullptr, nullptr, nullptr };
    for (int l = 0; l < c.n_layers; l++) {
        Qwen3Layer *         ly   = &m->layers[l];
        struct ggml_tensor * norm = qwen3_rms_norm(ctx, hidden, ly->input_layernorm, c.rms_norm_eps);
        struct ggml_tensor * attn =
            qw3lm_build_attn(ctx, gf, c, ly, norm, positions, mask, kv_rows, kv->k[kv_set][l], kv->v[kv_set][l],
                             n_kv_pad, n_tokens, m->use_flash_attn, m->clamp_fp16);
        hidden                   = ggml_add(ctx, hidden, attn);
        norm                     = qwen3_rms_norm(ctx, hidden, ly->post_attn_layernorm, c.rms_norm_eps);
        struct ggml_tensor * mlp = qwen3_build_mlp(ctx, ly, norm, n_tokens);
        hidden                   = ggml_add(ctx, hidden, mlp);
        for (int t = 0; t < 3; t++) {
            if (kTapLayers[t] == l) {
                taps[t] = hidden;
                ggml_set_output(hidden);
                ggml_build_forward_expand(gf, hidden);
            }
        }
    }
    hidden = qwen3_rms_norm(ctx, hidden, m->final_norm, c.rms_norm_eps);
    ggml_set_output(hidden);
    ggml_build_forward_expand(gf, hidden);
    struct ggml_tensor * last = ggml_cont(ctx, ggml_view_1d(ctx, hidden, H, (int64_t) (n_tokens - 1) * H * sizeof(float)));

    struct ggml_tensor * lgt[2];
    for (int p = 0; p < 2; p++) {
        int row0, rows;
        yue2_phase_rows(p == 0 ? YUE2_PHASE_ABC : YUE2_PHASE_SEMANTIC, &row0, &rows);
        lgt[p] = ggml_mul_mat(ctx, qw3lm_head_rows(ctx, m, row0, rows), last);
        ggml_set_output(lgt[p]);
        ggml_build_forward_expand(gf, lgt[p]);
    }

    if (g_exact) {
        int f32 = 0;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            struct ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->op == GGML_OP_MUL_MAT) {
                ggml_mul_mat_set_prec(n, GGML_PREC_F32);
                f32++;
            }
        }
        fprintf(stderr, "[Dump-AR] %d products at GGML_PREC_F32\n", f32);
    }

    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[Dump-AR] FATAL: failed to allocate the probe graph\n");
        exit(1);
    }
    ggml_backend_tensor_set(token_ids_t, token_ids, 0, n_tokens * sizeof(int));
    {
        std::vector<int>     pos_data(n_tokens);
        std::vector<int64_t> rows_data(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            pos_data[i]  = kv_pos + i;
            rows_data[i] = (int64_t) (kv_pos + i);
        }
        ggml_backend_tensor_set(positions, pos_data.data(), 0, n_tokens * sizeof(int));
        ggml_backend_tensor_set(kv_rows, rows_data.data(), 0, n_tokens * sizeof(int64_t));
    }
    {
        std::vector<uint16_t> mask_data((size_t) n_kv_pad * n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            int query_abs_pos = kv_pos + i;
            for (int j = 0; j < n_kv_pad; j++) {
                float v                              = (j <= query_abs_pos) ? 0.0f : -INFINITY;
                mask_data[(size_t) i * n_kv_pad + j] = ggml_fp32_to_fp16(v);
            }
        }
        ggml_backend_tensor_set(mask, mask_data.data(), 0, (size_t) n_kv_pad * n_tokens * sizeof(uint16_t));
    }
    ggml_backend_sched_graph_compute(m->sched, gf);

    size_t off = (size_t) (n_tokens - kTapRows) * H * sizeof(float);
    size_t len = (size_t) kTapRows * H * sizeof(float);
    for (int t = 0; t < 3; t++) {
        out->hidden[t].resize((size_t) kTapRows * H);
        ggml_backend_tensor_get(taps[t], out->hidden[t].data(), off, len);
    }
    out->normed.resize((size_t) kTapRows * H);
    ggml_backend_tensor_get(hidden, out->normed.data(), off, len);
    out->abc.resize(ggml_nelements(lgt[0]));
    out->semantic.resize(ggml_nelements(lgt[1]));
    ggml_backend_tensor_get(lgt[0], out->abc.data(), 0, out->abc.size() * sizeof(float));
    ggml_backend_tensor_get(lgt[1], out->semantic.data(), 0, out->semantic.size() * sizeof(float));

    kv->pos[kv_set] += n_tokens;
}

int main(int argc, char ** argv) {
    std::string model_path   = "/mnt/data/LLMs_models/Serveurperso/YuE2-GGUF/YuE2-3B-Q8_0.gguf";
    std::string request_path = "/mnt/data/golem-testdata/yue2/bench/guerreros-replay.json";
    std::string out_dir      = "/mnt/data/golem-testdata/yue2";
    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        request_path = argv[2];
    }
    if (argc > 3) {
        out_dir = argv[3];
    }
    if (argc > 4 && std::string(argv[4]) == "exact") {
        g_exact = true;
        setenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1", 1);
        setenv("GGML_VK_DISABLE_COOPMAT", "1", 1);
    }

    Yue2Request req;
    request_init(&req);
    if (!request_parse(&req, request_path.c_str())) {
        fprintf(stderr, "FATAL: failed to parse request %s\n", request_path.c_str());
        return 1;
    }
    BPETokenizer tok;
    if (!load_bpe_from_gguf(&tok, model_path.c_str())) {
        fprintf(stderr, "FATAL: cannot load BPE from %s\n", model_path.c_str());
        return 1;
    }
    auto encode = [&tok](const std::string & text) {
        return bpe_encode(&tok, text);
    };
    std::vector<int> abc_ids = encode(req.abc);
    std::vector<int> full    = yue2_build_prompt_ids(encode, YUE2_COT_FULL, req.style, req.lyrics, &abc_ids);
    std::vector<int> off     = yue2_build_prompt_ids(encode, YUE2_COT_OFF, req.style, req.lyrics, nullptr);
    std::vector<int> neg_off = yue2_build_negative_ids(encode, YUE2_COT_OFF, nullptr);
    fprintf(stderr, "[Dump-AR] full %zu, off %zu, negative %zu, lm_seed %lld\n", full.size(), off.size(),
            neg_off.size(), (long long) req.lm_seed);

    backend_set_device("Vulkan0");
    Qwen3LM lm;
    if (!qw3lm_load(&lm, model_path.c_str(), {})) {
        fprintf(stderr, "FATAL: failed to load LM from %s\n", model_path.c_str());
        return 1;
    }
    Qw3lmKvCache  kv;
    Qwen3LMConfig kv_cfg = lm.cfg;
    kv_cfg.max_seq_len   = 4096;
    qw3lm_kv_init(&kv, kv_cfg, lm.backend);
    if (!qw3lm_kv_sets(&kv, 2)) {
        fprintf(stderr, "FATAL: failed to allocate KV cache sets\n");
        return 1;
    }

    Fixtures fx(out_dir);
    const int H = lm.cfg.hidden_size;
    int       sem0, sem_rows, abc0, abc_rows;
    yue2_phase_rows(YUE2_PHASE_SEMANTIC, &sem0, &sem_rows);
    yue2_phase_rows(YUE2_PHASE_ABC, &abc0, &abc_rows);

    // 1. The prefill of the full prompt, tapped.
    const std::string dir = g_exact ? "ar_exact/" : "ar/";
    if (!g_exact) {
        fx.write_i32("ar/prefix", full.data(), { (int) full.size() });
    }
    PrefillProbes pp;
    qw3lm_kv_trim(&kv, 0, 0);
    probe_forward(&lm, &kv, full.data(), (int) full.size(), 0, &pp);
    for (int t = 0; t < 3; t++) {
        fx.write(dir + "prefill_hidden_l" + std::to_string(kTapLayers[t]), pp.hidden[t].data(), { kTapRows, H });
    }
    fx.write(dir + "prefill_normed", pp.normed.data(), { kTapRows, H });
    fx.write(dir + "prefill_logits_abc", pp.abc.data(), { abc_rows });
    fx.write(dir + "prefill_logits_semantic", pp.semantic.data(), { sem_rows });
    fprintf(stderr, "[Dump-AR] prefilled %zu tokens (pos %d)\n", full.size(), kv.pos[0]);
    if (g_exact) {
        qw3lm_kv_free(&kv);
        qw3lm_free(&lm);
        fprintf(stderr, "[Dump-AR] exact recording done\n");
        return 0;
    }

    // 2. Eight decode steps of the semantic phase, drawn as yue2_generate
    //    draws them: the draw index is the step, the seed the request's.
    {
        const int          steps  = 8;
        std::vector<float> logits = pp.semantic;
        std::vector<int>   history;
        std::vector<int>   tokens;
        std::vector<float> decoded;
        for (int step = 0; step < steps; step++) {
            std::vector<Yue2Candidate> cand;
            yue2_distribution(logits.data(), YUE2_SEMANTIC_SAMPLING, history, step, YUE2_PHASE_SEMANTIC, cand);
            int token = yue2_draw(cand, req.lm_seed, step);
            tokens.push_back(token);
            history.push_back(token);
            int set = 0;
            qw3lm_forward_batch(&lm, &kv, &token, &set, 1, logits.data(), sem0, sem_rows);
            decoded.insert(decoded.end(), logits.begin(), logits.end());
            fprintf(stderr, "[Dump-AR] decode step %d: %zu candidates, token %d\n", step, cand.size(), token);
        }
        fx.write_i32("ar/decode_tokens", tokens.data(), { steps });
        fx.write("ar/decode_logits", decoded.data(), { steps, sem_rows });
    }

    // 3. Cot off, guidance 1.01: both branches prefilled into sets 0 and 1,
    //    decoded side by side, mixed before the distribution.
    {
        const int   steps    = 4;
        const float guidance = 1.01f;
        fx.write_i32("ar/guided_prefix", off.data(), { (int) off.size() });
        fx.write_i32("ar/guided_negative", neg_off.data(), { (int) neg_off.size() });
        std::vector<float> batched((size_t) 2 * sem_rows);
        qw3lm_kv_trim(&kv, 0, 0);
        qw3lm_forward(&lm, &kv, off.data(), (int) off.size(), 0, batched.data(), sem0, sem_rows);
        qw3lm_kv_trim(&kv, 1, 0);
        qw3lm_forward(&lm, &kv, neg_off.data(), (int) neg_off.size(), 1, batched.data() + sem_rows, sem0, sem_rows);

        std::vector<float> cond, uncond, mixed(sem_rows);
        std::vector<int>   history, tokens;
        for (int step = 0; step < steps; step++) {
            const float * c = batched.data();
            const float * u = batched.data() + sem_rows;
            cond.insert(cond.end(), c, c + sem_rows);
            uncond.insert(uncond.end(), u, u + sem_rows);
            for (int k = 0; k < sem_rows; k++) {
                mixed[k] = u[k] + guidance * (c[k] - u[k]);
            }
            std::vector<Yue2Candidate> cand;
            yue2_distribution(mixed.data(), YUE2_SEMANTIC_SAMPLING, history, step, YUE2_PHASE_SEMANTIC, cand);
            int token = yue2_draw(cand, req.lm_seed, step);
            tokens.push_back(token);
            history.push_back(token);
            fprintf(stderr, "[Dump-AR] guided step %d: %zu candidates, token %d\n", step, cand.size(), token);
            if (step + 1 < steps) {
                int pair[2] = { token, token };
                int sets[2] = { 0, 1 };
                qw3lm_forward_batch(&lm, &kv, pair, sets, 2, batched.data(), sem0, sem_rows);
            }
        }
        fx.write("ar/guided_cond_logits", cond.data(), { steps, sem_rows });
        fx.write("ar/guided_uncond_logits", uncond.data(), { steps, sem_rows });
        fx.write_i32("ar/guided_tokens", tokens.data(), { steps });
    }

    qw3lm_kv_free(&kv);
    qw3lm_free(&lm);
    fprintf(stderr, "[Dump-AR] done\n");
    return 0;
}
