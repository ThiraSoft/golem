// Records the intermediate activations of a Gemma 4 forward pass, as llama.cpp
// computes them, so that the Go engine can be checked against them without
// llama.cpp being present at test time.
//
// llama.cpp names every waypoint of its graph ("attn_norm-0", "l_out-34",
// "result_norm"). A scheduler callback intercepts them by name and writes the
// float32 contents out. Two runs are produced: a short prompt where every
// waypoint of four representative blocks is kept, and a long one that exists
// only to push past the 512-position window.

#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

struct dumped {
    std::string file;
    int64_t ne[4];
};

struct dump_state {
    std::string          dir;
    std::set<std::string> wanted;
    std::map<std::string, dumped> written;
    bool                 active = true;
    // when set, only the last column of dimension 1 is kept
    bool                 last_column_only = false;
};

static void die(const char * what) {
    fprintf(stderr, "dump_layers: %s\n", what);
    exit(1);
}

static void write_floats(const std::string & path, const std::vector<float> & v) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) die(("cannot write " + path).c_str());
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}

static bool on_node(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    if (!st->active) return false;

    const std::string name = ggml_get_name(t);
    if (st->wanted.find(name) == st->wanted.end()) return false;
    if (ask) return true;   // yes, please compute this one and call me back

    if (t->type != GGML_TYPE_F32) die((name + " is not F32").c_str());
    if (!ggml_is_contiguous(t))   die((name + " is not contiguous").c_str());

    const int64_t n0 = t->ne[0];
    const int64_t n1 = t->ne[1];
    const int64_t rest = t->ne[2] * t->ne[3];

    std::vector<float> all(ggml_nelements(t));
    ggml_backend_tensor_get(t, all.data(), 0, ggml_nbytes(t));

    dumped d;
    d.file = name + ".bin";
    if (st->last_column_only && n1 > 1) {
        // keep column n1-1 of every slice of the higher dimensions
        std::vector<float> tail;
        tail.reserve(n0 * rest);
        for (int64_t r = 0; r < rest; ++r) {
            const float * slice = all.data() + r * n0 * n1;
            tail.insert(tail.end(), slice + (n1 - 1) * n0, slice + n1 * n0);
        }
        write_floats(st->dir + "/" + d.file, tail);
        d.ne[0] = n0; d.ne[1] = 1; d.ne[2] = t->ne[2]; d.ne[3] = t->ne[3];
    } else {
        write_floats(st->dir + "/" + d.file, all);
        for (int i = 0; i < 4; ++i) d.ne[i] = t->ne[i];
    }
    st->written[name] = d;
    return true;
}

// The waypoints kept for the short run. Seven blocks cover every kind and
// every pairing: in E2B, 0 and 13 own window caches, 4 and 14 own global ones,
// 15 reads block 13's, 19 reads block 14's; blocks 13 and 14 are kept because
// the sharing tests need the source's input in order to fill the cache the
// sharer then reads, without running the whole stack first. Block 5 is there
// for the 12B, where it is the first global block — the kind that publishes no
// value projection and uses its keys as values.
//
// The two checkpoints do not agree on which of those blocks compute keys at
// all, and only E2B has per-layer embeddings, so the waypoints that depend on
// either are asked for and allowed to be absent rather than being predicted
// from the block number.
static std::set<std::string> short_run_names(int n_layer) {
    std::set<std::string> names = { "inp_scaled", "result_norm" };
    for (int il : {0, 4, 5, 13, 14, 15, 19}) {
        if (il >= n_layer) continue;
        const std::string s = "-" + std::to_string(il);
        names.insert("attn_norm"          + s);
        names.insert("Qcur_normed"        + s);
        names.insert("Qcur_pos"           + s);
        names.insert("kqv_out"            + s);
        names.insert("attn_post_norm"     + s);
        names.insert("attn_out"           + s);
        names.insert("ffn_norm"           + s);
        names.insert("ffn_out"            + s);
        names.insert("ffn_post_norm"      + s);
        names.insert("l_out"              + s);
    }
    // Every block's output, for all thirty-five: it is what the next block
    // reads, so a test can start any block from the reference's own input
    // instead of running everything before it, and a divergence in the full
    // stack says which block it began in.
    for (int il = 0; il < 64; ++il) {
        names.insert("l_out-" + std::to_string(il));
    }
    return names;
}

// The waypoints a checkpoint may or may not have: the keys and values of a
// block that shares another's cache do not exist, and neither does anything
// per-layer in a model that declares no per-layer width.
static std::set<std::string> optional_names(int n_layer) {
    std::set<std::string> names = { "per_layer_proj", "inp_per_layer" };
    for (int il : {0, 4, 5, 13, 14, 15, 19}) {
        if (il >= n_layer) continue;
        const std::string s = "-" + std::to_string(il);
        names.insert("Kcur_normed"        + s);
        names.insert("Kcur_pos"           + s);
        names.insert("Vcur_normed"        + s);
        names.insert("pe_in"              + s);
        names.insert("per_layer_embd_out" + s);
    }
    return names;
}

// The block count is not known before the model is loaded, so short_run_names
// asks for more l_out entries than exist; prune the ones the graph never had.
static void prune_absent_l_out(std::set<std::string> & names, int n_layer) {
    for (int il = n_layer; il < 64; ++il) {
        names.erase("l_out-" + std::to_string(il));
    }
}

static void write_index(const dump_state & st,
                        const std::string & prompt,
                        const std::string & model_name,
                        const std::vector<llama_token> & tokens,
                        const std::vector<std::pair<int, float>> & top,
                        const std::vector<llama_token> & greedy,
                        int n_embd, int n_layer, int n_embd_per_layer) {
    const std::string path = st.dir + "/index.json";
    FILE * f = fopen(path.c_str(), "w");
    if (!f) die(("cannot write " + path).c_str());

    fprintf(f, "{\n");
    fprintf(f, "  \"model\": \"%s\",\n", model_name.c_str());
    fprintf(f, "  \"prompt\": \"%s\",\n", prompt.c_str());
    fprintf(f, "  \"n_embd\": %d,\n  \"n_layer\": %d,\n  \"n_embd_per_layer\": %d,\n",
            n_embd, n_layer, n_embd_per_layer);

    fprintf(f, "  \"tokens\": [");
    for (size_t i = 0; i < tokens.size(); ++i) fprintf(f, "%s%d", i ? ", " : "", tokens[i]);
    fprintf(f, "],\n");

    fprintf(f, "  \"tensors\": {\n");
    size_t i = 0;
    for (const auto & kv : st.written) {
        fprintf(f, "    \"%s\": {\"file\": \"%s\", \"ne\": [%lld, %lld, %lld, %lld]}%s\n",
                kv.first.c_str(), kv.second.file.c_str(),
                (long long) kv.second.ne[0], (long long) kv.second.ne[1],
                (long long) kv.second.ne[2], (long long) kv.second.ne[3],
                ++i == st.written.size() ? "" : ",");
    }
    fprintf(f, "  },\n");

    fprintf(f, "  \"logits_top\": [");
    for (size_t j = 0; j < top.size(); ++j) {
        fprintf(f, "%s{\"id\": %d, \"logit\": %.9g}", j ? ", " : "", top[j].first, top[j].second);
    }
    fprintf(f, "],\n");
    fprintf(f, "  \"argmax\": %d,\n", top.empty() ? -1 : top[0].first);

    fprintf(f, "  \"greedy\": [");
    for (size_t j = 0; j < greedy.size(); ++j) fprintf(f, "%s%d", j ? ", " : "", greedy[j]);
    fprintf(f, "]\n}\n");
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: dump_layers <model.gguf> <out-dir> <short|window>\n");
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string out_dir    = argv[2];
    const std::string mode       = argv[3];
    if (mode != "short" && mode != "window") die("mode must be short or window");

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;                 // the CPU backend is the reference
    // An empty device list keeps every graph on the CPU. Without it llama.cpp
    // picks up whatever accelerator the machine has, and the recordings would
    // then be that accelerator's arithmetic rather than ggml's reference one.
    static ggml_backend_dev_t no_devices[] = { nullptr };
    mparams.devices = no_devices;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) die("cannot load the model");

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // A short factual prompt for the parity run; the same sentence repeated
    // until it passes 512 positions for the window run.
    std::string prompt = "The capital of France is";
    if (mode == "window") {
        std::string one = " The capital of France is Paris and the capital of Japan is Tokyo.";
        prompt = "";
        for (int i = 0; i < 60; ++i) prompt += one;
    }

    std::vector<llama_token> tokens(4096);
    int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                           tokens.data(), (int32_t) tokens.size(), true, false);
    if (n <= 0) die("tokenization failed");
    tokens.resize(n);
    if (mode == "window" && n <= 512) die("the window run must exceed 512 tokens");
    fprintf(stderr, "dump_layers: %d tokens\n", n);

    dump_state st;
    st.dir = out_dir;
    std::set<std::string> optional;
    if (mode == "short") {
        st.wanted = short_run_names(llama_model_n_layer(model));
        prune_absent_l_out(st.wanted, llama_model_n_layer(model));
        optional = optional_names(llama_model_n_layer(model));
        st.wanted.insert(optional.begin(), optional.end());
    } else {
        st.wanted = {"l_out-0", "l_out-4", "l_out-15", "result_norm"};
        st.last_column_only = true;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = 4096;
    cparams.n_batch           = 4096;
    cparams.n_ubatch          = 4096;   // one graph, so each name fires once
    cparams.n_threads         = 8;
    cparams.n_threads_batch   = 8;
    // Flash attention fuses the waypoints away; the graph has to stay legible.
    cparams.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval           = on_node;
    cparams.cb_eval_user_data = &st;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) die("cannot create the context");

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n))) die("decode failed");

    st.active = false;   // the greedy continuation below must not overwrite

    for (const auto & name : st.wanted) {
        if (st.written.find(name) != st.written.end()) continue;
        if (optional.find(name) != optional.end()) continue;
        fprintf(stderr, "dump_layers: %s never appeared in the graph\n", name.c_str());
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);

    std::vector<std::pair<int, float>> top;
    top.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) top.emplace_back(i, logits[i]);
    std::partial_sort(top.begin(), top.begin() + 64, top.end(),
                      [](const auto & a, const auto & b) { return a.second > b.second; });
    top.resize(64);

    // Sixteen greedy steps, which is what the end-to-end test replays.
    std::vector<llama_token> greedy;
    llama_token next = top[0].first;
    for (int step = 0; step < 16; ++step) {
        greedy.push_back(next);
        if (llama_decode(ctx, llama_batch_get_one(&next, 1))) die("decode failed");
        const float * l = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (l[i] > l[best]) best = i;
        next = best;
    }

    const size_t slash = model_path.find_last_of('/');
    write_index(st, mode == "window" ? "(repeated sentence)" : prompt,
                slash == std::string::npos ? model_path : model_path.substr(slash + 1),
                tokens, top, greedy,
                llama_model_n_embd(model), llama_model_n_layer(model),
                st.written.count("inp_per_layer")
                    ? (int) st.written["inp_per_layer"].ne[0] : 0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "dump_layers: wrote %zu tensors to %s\n", st.written.size(), out_dir.c_str());
    return 0;
}
