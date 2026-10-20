#include "backend.h"
#include "bpe.h"
#include "dump_common.h"
#include "philox.h"
#include "prompt.h"
#include "qwen3-lm.h"
#include "request.h"
#include "sampling.h"
#include "unicode.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Full canonical decomposition followed by canonical ordering of combining marks.
static std::string uni_nfd(const std::string & text) {
    bool ascii = true;
    for (size_t i = 0; i < text.size(); i++) {
        if ((unsigned char) text[i] >= 0x80) {
            ascii = false;
            break;
        }
    }
    if (ascii) {
        return text;
    }

    std::vector<uint32_t> seq;
    seq.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        int adv = 0;
        uni_decompose((uint32_t) utf8_codepoint(text.c_str() + i, &adv), seq);
        i += (size_t) adv;
    }

    // Canonical ordering: combining marks sort by class, stable inside a run
    for (size_t i = 1; i < seq.size(); i++) {
        int cc = uni_ccc(seq[i]);
        if (cc == 0) {
            continue;
        }
        size_t j = i;
        while (j > 0 && uni_ccc(seq[j - 1]) > cc) {
            uint32_t tmp = seq[j - 1];
            seq[j - 1]   = seq[j];
            seq[j]       = tmp;
            j--;
        }
    }

    std::string out;
    out.reserve(text.size() * 2);
    for (size_t i = 0; i < seq.size(); i++) {
        utf8_append(out, seq[i]);
    }
    return out;
}

static void record_sample_case(Fixtures & fixtures,
                               const std::string & prefix,
                               const float * logits, int rows,
                               const std::vector<int> & history,
                               int step,
                               const std::vector<Yue2Candidate> & candidates,
                               int token) {
    fixtures.write("sample/" + prefix + "_logits", logits, {rows});
    fixtures.write_i32("sample/" + prefix + "_history", history.data(), {(int)history.size()});
    std::vector<int> cand_ids(candidates.size());
    std::vector<float> cand_scores(candidates.size());
    for (size_t i = 0; i < candidates.size(); i++) {
        cand_ids[i] = candidates[i].id;
        cand_scores[i] = candidates[i].score;
    }
    fixtures.write_i32("sample/" + prefix + "_cand_ids", cand_ids.data(), {(int)cand_ids.size()});
    fixtures.write("sample/" + prefix + "_cand_scores", cand_scores.data(), {(int)cand_scores.size()});
    int32_t tok = token;
    fixtures.write_i32("sample/" + prefix + "_token", &tok, {1});
    int32_t st = step;
    fixtures.write_i32("sample/" + prefix + "_step", &st, {1});
}

int main(int argc, char ** argv) {
    std::string model_path    = "/mnt/data/LLMs_models/Serveurperso/YuE2-GGUF/YuE2-3B-Q8_0.gguf";
    std::string request_path  = "/mnt/data/golem-testdata/yue2/bench/guerreros-replay.json";
    std::string bpe_text_path = "/mnt/data/dev/musicbench/yue2/tests/bpe-text.txt";
    std::string out_dir       = "/mnt/data/golem-testdata/yue2";

    if (argc > 1) model_path    = argv[1];
    if (argc > 2) request_path  = argv[2];
    if (argc > 3) bpe_text_path = argv[3];
    if (argc > 4) out_dir       = argv[4];

    fprintf(stderr, "[Dump-Tokens] Starting reference token/prompt/sampler dumper\n");
    fprintf(stderr, "  Backbone: %s\n", model_path.c_str());
    fprintf(stderr, "  Request:  %s\n", request_path.c_str());
    fprintf(stderr, "  BPE text: %s\n", bpe_text_path.c_str());
    fprintf(stderr, "  Output:   %s\n", out_dir.c_str());

    Fixtures fixtures(out_dir);

    // 1. Load BPE tokenizer from GGUF
    BPETokenizer tok;
    if (!load_bpe_from_gguf(&tok, model_path.c_str())) {
        fprintf(stderr, "FATAL: cannot load BPE from %s\n", model_path.c_str());
        return 1;
    }
    auto encode = [&tok](const std::string & text) {
        return bpe_encode(&tok, text);
    };

    // 2. Parse request JSON
    Yue2Request req;
    request_init(&req);
    if (!request_parse(&req, request_path.c_str())) {
        fprintf(stderr, "FATAL: failed to parse request %s\n", request_path.c_str());
        return 1;
    }

    // 3. Read BPE test text file
    std::ifstream bpe_in(bpe_text_path, std::ios::binary);
    if (!bpe_in.is_open()) {
        fprintf(stderr, "FATAL: cannot open %s\n", bpe_text_path.c_str());
        return 1;
    }
    std::string bpe_text((std::istreambuf_iterator<char>(bpe_in)), std::istreambuf_iterator<char>());
    bpe_in.close();

    // 4. Tokenizer fixtures under tokens/
    // (a) Guerreros request text
    std::string req_text = yue2_request_text(YUE2_COT_FULL, req.style, req.lyrics);
    std::vector<int> req_ids = bpe_encode(&tok, req_text);
    fixtures.write_i32("tokens/guerreros_request", req_ids.data(), {(int)req_ids.size()});

    // (b) bpe-text.txt
    std::vector<int> bpe_text_ids = bpe_encode(&tok, bpe_text);
    fixtures.write_i32("tokens/bpe_text", bpe_text_ids.data(), {(int)bpe_text_ids.size()});

    // (c) Guerreros lyrics in NFD
    std::string nfd_lyrics = uni_nfd(req.lyrics);
    std::vector<int> nfd_lyrics_ids = bpe_encode(&tok, nfd_lyrics);
    fixtures.write_i32("tokens/guerreros_lyrics_nfd", nfd_lyrics_ids.data(), {(int)nfd_lyrics_ids.size()});

    // (d) Guerreros replay ABC
    std::vector<int> abc_ids = bpe_encode(&tok, req.abc);
    fixtures.write_i32("tokens/guerreros_abc", abc_ids.data(), {(int)abc_ids.size()});

    fprintf(stderr, "[Dump-Tokens] Wrote BPE fixtures: request (%zu), bpe_text (%zu), nfd_lyrics (%zu), abc (%zu)\n",
            req_ids.size(), bpe_text_ids.size(), nfd_lyrics_ids.size(), abc_ids.size());

    // 5. Prompt fixtures under prompt/
    // (a) full with ABC
    std::vector<int> p_full = yue2_build_prompt_ids(encode, YUE2_COT_FULL, req.style, req.lyrics, &abc_ids);
    fixtures.write_i32("prompt/guerreros_full_with_abc", p_full.data(), {(int)p_full.size()});

    // (b) melody without ABC
    std::vector<int> p_melody = yue2_build_prompt_ids(encode, YUE2_COT_MELODY, req.style, req.lyrics, nullptr);
    fixtures.write_i32("prompt/guerreros_melody_no_abc", p_melody.data(), {(int)p_melody.size()});

    // (c) cot off
    std::vector<int> p_off = yue2_build_prompt_ids(encode, YUE2_COT_OFF, req.style, req.lyrics, nullptr);
    fixtures.write_i32("prompt/guerreros_off", p_off.data(), {(int)p_off.size()});

    // (d) negative off
    std::vector<int> neg_off = yue2_build_negative_ids(encode, YUE2_COT_OFF, nullptr);
    fixtures.write_i32("prompt/negative_off", neg_off.data(), {(int)neg_off.size()});

    // (e) negative full with ABC
    std::vector<int> neg_full = yue2_build_negative_ids(encode, YUE2_COT_FULL, &abc_ids);
    fixtures.write_i32("prompt/negative_full_with_abc", neg_full.data(), {(int)neg_full.size()});

    fprintf(stderr, "[Dump-Tokens] Wrote prompt fixtures: full (%zu), melody (%zu), off (%zu), neg_off (%zu), neg_full (%zu)\n",
            p_full.size(), p_melody.size(), p_off.size(), neg_off.size(), neg_full.size());

    // 6. Philox test cases under philox/
    struct PhiloxTestCase {
        uint32_t ctr[4];
        uint32_t seed_lo;
        uint32_t seed_hi;
    };
    std::vector<PhiloxTestCase> philox_cases = {
        { {0, 0, 0, 0}, 0, 0 },
        { {1, 0, 0, 0}, 1, 0 },
        { {0, 0, 151847, 0}, (uint32_t)req.lm_seed, (uint32_t)((uint64_t)req.lm_seed >> 32) },
        { {42, 1, 151853, 0}, (uint32_t)req.lm_seed, (uint32_t)((uint64_t)req.lm_seed >> 32) },
        { {1000, 5, 25000, 0}, (uint32_t)req.lm_seed, (uint32_t)((uint64_t)req.lm_seed >> 32) },
        { {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}, 0x12345678u, 0x9ABCDEF0u },
        { {7, 0, 100, 0}, 0xCAFEFEEDu, 0xDEADBEEFu },
        { {1234567, 89, 54321, 0}, 0x87654321u, 0xFEDCBA98u },
    };
    int N_philox = (int)philox_cases.size();
    std::vector<int32_t> philox_counters((size_t)N_philox * 4);
    std::vector<int32_t> philox_seeds((size_t)N_philox * 2);
    std::vector<int32_t> philox_words((size_t)N_philox * 4);
    for (int i = 0; i < N_philox; i++) {
        const auto & tc = philox_cases[i];
        for (int k = 0; k < 4; k++) philox_counters[i * 4 + k] = (int32_t)tc.ctr[k];
        philox_seeds[i * 2 + 0] = (int32_t)tc.seed_lo;
        philox_seeds[i * 2 + 1] = (int32_t)tc.seed_hi;
        Philox4 r = philox4x32_10({tc.ctr[0], tc.ctr[1], tc.ctr[2], tc.ctr[3]}, tc.seed_lo, tc.seed_hi);
        philox_words[i * 4 + 0] = (int32_t)r.x;
        philox_words[i * 4 + 1] = (int32_t)r.y;
        philox_words[i * 4 + 2] = (int32_t)r.z;
        philox_words[i * 4 + 3] = (int32_t)r.w;
    }
    fixtures.write_i32("philox/counters", philox_counters.data(), {N_philox, 4});
    fixtures.write_i32("philox/seeds", philox_seeds.data(), {N_philox, 2});
    fixtures.write_i32("philox/words", philox_words.data(), {N_philox, 4});
    fprintf(stderr, "[Dump-Tokens] Wrote Philox fixtures: %d cases\n", N_philox);

    // 7. Sampler cases on Vulkan0
    backend_set_device("Vulkan0");
    Qwen3LM lm;
    if (!qw3lm_load(&lm, model_path.c_str(), {})) {
        fprintf(stderr, "FATAL: failed to load LM from %s\n", model_path.c_str());
        return 1;
    }

    Qw3lmKvCache kv;
    Qwen3LMConfig kv_cfg = lm.cfg;
    kv_cfg.max_seq_len   = 4096;
    qw3lm_kv_init(&kv, kv_cfg, lm.backend);
    if (!qw3lm_kv_sets(&kv, 1)) {
        fprintf(stderr, "FATAL: failed to allocate KV cache sets\n");
        return 1;
    }

    // 7a. Score phase: 3 steps from open prefix (cot full, no abc)
    std::vector<int> p_score = yue2_build_prompt_ids(encode, YUE2_COT_FULL, req.style, req.lyrics, nullptr);
    int score_row0, score_rows;
    yue2_phase_rows(YUE2_PHASE_ABC, &score_row0, &score_rows);
    std::vector<float> score_logits(score_rows);

    fprintf(stderr, "[Dump-Tokens] Prefilling score prefix (%zu tokens) on Vulkan0...\n", p_score.size());
    qw3lm_kv_trim(&kv, 0, 0);
    qw3lm_forward(&lm, &kv, p_score.data(), (int)p_score.size(), 0, score_logits.data(), score_row0, score_rows);
    fprintf(stderr, "[Dump-Tokens] Score prefix prefilled (pos = %d)\n", kv.pos[0]);

    std::vector<float> initial_score_logits = score_logits;
    std::vector<int> score_history;
    for (int step = 0; step < 3; step++) {
        std::vector<Yue2Candidate> candidates;
        yue2_distribution(score_logits.data(), YUE2_ABC_SAMPLING, score_history, step, YUE2_PHASE_ABC, candidates);
        int token = yue2_draw(candidates, req.lm_seed, step);

        std::string case_name = "score_" + std::to_string(step);
        record_sample_case(fixtures, case_name, score_logits.data(), score_rows, score_history, step, candidates, token);
        fprintf(stderr, "[Dump-Tokens] Recorded sample/%s: %zu candidates, drawn token %d\n",
                case_name.c_str(), candidates.size(), token);

        int t_arr[1] = { token };
        int s_arr[1] = { 0 };
        qw3lm_forward_batch(&lm, &kv, t_arr, s_arr, 1, score_logits.data(), score_row0, score_rows);
        score_history.push_back(token);
    }

    // 7b. Semantic phase: 3 steps from full prefix with replay's ABC
    std::vector<int> p_sem = yue2_build_prompt_ids(encode, YUE2_COT_FULL, req.style, req.lyrics, &abc_ids);
    int sem_row0, sem_rows;
    yue2_phase_rows(YUE2_PHASE_SEMANTIC, &sem_row0, &sem_rows);
    std::vector<float> sem_logits(sem_rows);

    fprintf(stderr, "[Dump-Tokens] Prefilling semantic prefix (%zu tokens) on Vulkan0...\n", p_sem.size());
    qw3lm_kv_trim(&kv, 0, 0);
    qw3lm_forward(&lm, &kv, p_sem.data(), (int)p_sem.size(), 0, sem_logits.data(), sem_row0, sem_rows);
    fprintf(stderr, "[Dump-Tokens] Semantic prefix prefilled (pos = %d)\n", kv.pos[0]);

    std::vector<float> initial_sem_logits = sem_logits;
    std::vector<int> sem_history;
    for (int step = 0; step < 3; step++) {
        std::vector<Yue2Candidate> candidates;
        yue2_distribution(sem_logits.data(), YUE2_SEMANTIC_SAMPLING, sem_history, step, YUE2_PHASE_SEMANTIC, candidates);
        int token = yue2_draw(candidates, req.lm_seed, step);

        std::string case_name = "semantic_" + std::to_string(step);
        record_sample_case(fixtures, case_name, sem_logits.data(), sem_rows, sem_history, step, candidates, token);
        fprintf(stderr, "[Dump-Tokens] Recorded sample/%s: %zu candidates, drawn token %d\n",
                case_name.c_str(), candidates.size(), token);

        int t_arr[1] = { token };
        int s_arr[1] = { 0 };
        qw3lm_forward_batch(&lm, &kv, t_arr, s_arr, 1, sem_logits.data(), sem_row0, sem_rows);
        sem_history.push_back(token);
    }

    // 7c. Penalty window case: history longer than penalty window (50) with repeated ids
    std::vector<int> synth_history;
    for (int i = 0; i < 70; i++) {
        int id = YUE2_CODEC_OFFSET + (i % 20); // repeated IDs within [151853, 151872]
        synth_history.push_back(id);
    }
    std::vector<Yue2Candidate> penalty_candidates;
    yue2_distribution(initial_sem_logits.data(), YUE2_SEMANTIC_SAMPLING, synth_history, 0, YUE2_PHASE_SEMANTIC, penalty_candidates);
    int penalty_token = yue2_draw(penalty_candidates, req.lm_seed, 0);
    record_sample_case(fixtures, "penalty_window", initial_sem_logits.data(), sem_rows, synth_history, 0, penalty_candidates, penalty_token);
    fprintf(stderr, "[Dump-Tokens] Recorded sample/penalty_window: %zu candidates, drawn token %d\n",
            penalty_candidates.size(), penalty_token);

    // 7d. Min tokens case: step >= min_tokens so end token (ABC_END) is present
    std::vector<int> empty_history;
    std::vector<Yue2Candidate> end_token_candidates;
    yue2_distribution(initial_score_logits.data(), YUE2_ABC_SAMPLING, empty_history, 35, YUE2_PHASE_ABC, end_token_candidates);
    int end_token_draw = yue2_draw(end_token_candidates, req.lm_seed, 35);
    record_sample_case(fixtures, "with_end_token", initial_score_logits.data(), score_rows, empty_history, 35, end_token_candidates, end_token_draw);
    fprintf(stderr, "[Dump-Tokens] Recorded sample/with_end_token: %zu candidates, drawn token %d\n",
            end_token_candidates.size(), end_token_draw);

    qw3lm_kv_free(&kv);
    fprintf(stderr, "[Dump-Tokens] Completed successfully\n");
    return 0;
}
