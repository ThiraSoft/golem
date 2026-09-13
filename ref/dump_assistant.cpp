// Records one draft by a Gemma 4 assistant, the way llama.cpp's draft-mtp
// speculation makes it, so that the Go drafter can be checked against it
// waypoint by waypoint.
//
// The assistant has no cache of its own: its context is created with the
// target's as ctx_other, and its four blocks read the target's last window and
// last global block. So the target reads the prompt first, and the draft is
// the token after the prompt's greedy continuation — the same pairing
// common/speculative.cpp builds: the token just decided, at the position it is
// about to occupy, with the target's normed state from the position before.
//
// Then a greedy run of `steps` tokens, recording at each the target's choice
// and the assistant's guess for the token after it. That is what the
// end-to-end test replays.
//
// The run file takes dump_layers' directives, less the block list: every block
// of the assistant is recorded, there are only four. `steps <n>` is new.

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

struct dumped {
    std::string file;
    int64_t ne[4];
};

struct dump_state {
    std::string                   dir;
    std::set<std::string>         wanted;
    std::map<std::string, dumped> written;
    bool                          active = true;
};

static void die(const char * what) {
    fprintf(stderr, "dump_assistant: %s\n", what);
    exit(1);
}

static void write_floats(const std::string & path, const float * v, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) die(("cannot write " + path).c_str());
    fwrite(v, sizeof(float), n, f);
    fclose(f);
}

static bool on_node(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    if (!st->active) return false;

    const std::string name = ggml_get_name(t);
    if (st->wanted.find(name) == st->wanted.end()) return false;
    if (ask) return true;

    if (t->type != GGML_TYPE_F32) die((name + " is not F32").c_str());
    if (!ggml_is_contiguous(t))   die((name + " is not contiguous").c_str());

    std::vector<float> all(ggml_nelements(t));
    ggml_backend_tensor_get(t, all.data(), 0, ggml_nbytes(t));

    dumped d;
    d.file = name + ".bin";
    write_floats(st->dir + "/" + d.file, all.data(), all.size());
    for (int i = 0; i < 4; ++i) d.ne[i] = t->ne[i];
    st->written[name] = d;
    return true;
}

struct run_spec {
    std::string prompt;
    std::string label;
    bool add_special   = true;
    bool parse_special = false;
    int  steps         = 16;
    std::vector<std::string> target_names;
    std::vector<std::string> require_names;
    std::vector<std::string> global_names;
    std::vector<std::string> global_opt;
};

static std::string unescape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 == s.size()) { out += s[i]; continue; }
        switch (s[++i]) {
            case 'n':  out += '\n'; break;
            case 's':  out += ' ';  break;
            case '\\': out += '\\'; break;
            default:   die("unknown escape in a run file");
        }
    }
    return out;
}

static run_spec read_run(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) die(("cannot read " + path).c_str());

    run_spec r;
    bool have_label = false;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
        if (s.empty() || s[0] == '#') continue;

        const size_t sp  = s.find(' ');
        const std::string key = s.substr(0, sp);
        std::string val = sp == std::string::npos ? "" : s.substr(sp + 1);
        const size_t first = val.find_first_not_of(' ');
        val = first == std::string::npos ? "" : val.substr(first);

        if      (key == "prompt")      r.prompt = unescape(val);
        else if (key == "label")     { r.label  = unescape(val); have_label = true; }
        else if (key == "add_special") r.add_special = atoi(val.c_str()) != 0;
        else if (key == "parse_special") r.parse_special = atoi(val.c_str()) != 0;
        else if (key == "steps")       r.steps = atoi(val.c_str());
        else if (key == "target")      r.target_names.push_back(val);
        else if (key == "require")     r.require_names.push_back(val);
        else if (key == "global")      r.global_names.push_back(val);
        else if (key == "global_opt")  r.global_opt.push_back(val);
        else die(("unknown directive: " + key).c_str());
    }
    fclose(f);

    if (r.prompt.empty()) die("the run file declares no prompt");
    if (!have_label) r.label = r.prompt;
    return r;
}

static llama_token argmax(const float * logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (logits[i] > logits[best]) best = i;
    return best;
}

static std::string base_name(const std::string & path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// A chat prompt carries newlines and markers, which a JSON string cannot hold
// as they are.
static std::string json_escape(const std::string & s) {
    std::string out;
    for (const char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static void write_ids(FILE * f, const char * key, const std::vector<llama_token> & ids, bool last) {
    fprintf(f, "  \"%s\": [", key);
    for (size_t i = 0; i < ids.size(); ++i) fprintf(f, "%s%d", i ? ", " : "", ids[i]);
    fprintf(f, "]%s\n", last ? "" : ",");
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: dump_assistant <target.gguf> <assistant.gguf> <out-dir> <run-file>\n");
        return 1;
    }
    const std::string target_path    = argv[1];
    const std::string assistant_path = argv[2];
    const std::string out_dir        = argv[3];
    const run_spec    run            = read_run(argv[4]);

    llama_backend_init();

    // The CPU backend is the reference, for both: an empty device list keeps
    // every graph off whatever accelerator the machine has.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    static ggml_backend_dev_t no_devices[] = { nullptr };
    mparams.devices = no_devices;

    llama_model * target = llama_model_load_from_file(target_path.c_str(), mparams);
    if (!target) die("cannot load the target");
    llama_model * assistant = llama_model_load_from_file(assistant_path.c_str(), mparams);
    if (!assistant) die("cannot load the assistant");

    const llama_vocab * vocab = llama_model_get_vocab(target);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(8192);
    const int n = llama_tokenize(vocab, run.prompt.c_str(), (int32_t) run.prompt.size(),
                                 tokens.data(), (int32_t) tokens.size(), run.add_special, run.parse_special);
    if (n <= 0) die("tokenization failed");
    tokens.resize(n);
    fprintf(stderr, "dump_assistant: %d tokens\n", n);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 2048;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_threads       = 8;
    cparams.n_threads_batch = 8;
    // Flash attention fuses the waypoints away, and the two contexts must
    // agree on it: the assistant reads the target's values in the target's
    // layout.
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    // The target's own waypoints, over the whole prompt: the input of the
    // blocks whose caches the assistant reads, so that the Go side can fill
    // those caches from the reference rather than from its own 46 blocks.
    dump_state tst;
    tst.dir = out_dir;
    for (const auto & name : run.target_names) tst.wanted.insert(name);
    cparams.cb_eval           = on_node;
    cparams.cb_eval_user_data = &tst;

    llama_context * ctx_tgt = llama_init_from_model(target, cparams);
    if (!ctx_tgt) die("cannot create the target's context");
    // Unmasked: a row for every position of a batch, which is what
    // common/speculative.cpp asks of the target.
    llama_set_embeddings_nextn(ctx_tgt, true, false);

    dump_state st;
    st.dir = out_dir;
    std::set<std::string> optional;
    for (const auto & name : run.global_names) st.wanted.insert(name);
    for (const auto & name : run.global_opt) { st.wanted.insert(name); optional.insert(name); }
    // The assistant's blocks are prediction layers, and llama.cpp counts them
    // apart: its n_layer is zero.
    const int n_layer = llama_model_n_layer_nextn(assistant);
    if (n_layer <= 0) die("the assistant declares no blocks");
    for (int il = 0; il < n_layer; ++il) {
        for (const auto & name : run.require_names) st.wanted.insert(name + "-" + std::to_string(il));
    }

    llama_context_params dparams = cparams;
    dparams.n_batch           = 8;
    dparams.n_ubatch          = 8;
    dparams.ctx_type          = LLAMA_CONTEXT_TYPE_MTP;
    dparams.ctx_other         = ctx_tgt;
    dparams.cb_eval           = on_node;
    dparams.cb_eval_user_data = &st;
    llama_context * ctx_dft = llama_init_from_model(assistant, dparams);
    if (!ctx_dft) die("cannot create the assistant's context");
    llama_set_embeddings_nextn(ctx_dft, true, true);

    const int n_embd_out = llama_model_n_embd_out(assistant);
    if (n_embd_out != llama_model_n_embd(target)) die("the assistant does not take the target's width");

    // The draft batch carries a token and a row both, which llama_batch_init
    // does not allocate together; common/speculative.cpp mallocs the token
    // array the same way.
    llama_batch draft = llama_batch_init(1, n_embd_out, 1);
    draft.token = (llama_token *) malloc(sizeof(llama_token));

    std::vector<float> h(n_embd_out);
    auto guess_from = [&](llama_token id, int pos, const float * row) -> llama_token {
        draft.n_tokens     = 1;
        draft.token[0]     = id;
        draft.pos[0]       = pos;
        draft.n_seq_id[0]  = 1;
        draft.seq_id[0][0] = 0;
        draft.logits[0]    = 1;
        std::memcpy(draft.embd, row, h.size() * sizeof(float));
        if (llama_decode(ctx_dft, draft)) die("the assistant's decode failed");
        return argmax(llama_get_logits_ith(ctx_dft, 0), n_vocab);
    };
    auto guess = [&](llama_token id, int pos) { return guess_from(id, pos, h.data()); };

    // The second guess of a step, which is how draft-mtp drafts two: the first
    // guess fed back with the assistant's own projected state, at the same
    // position — the cache is shared and nothing was written at it
    // (common/speculative.cpp, the is_mem_shared branch of draft()).
    std::vector<float> own(n_embd_out);
    auto second = [&](llama_token first, int pos) -> llama_token {
        std::memcpy(own.data(), llama_get_embeddings_nextn_ith(ctx_dft, 0), own.size() * sizeof(float));
        return guess_from(first, pos, own.data());
    };

    if (llama_decode(ctx_tgt, llama_batch_get_one(tokens.data(), n))) die("the target's decode failed");
    tst.active = false;
    for (const auto & name : tst.wanted) {
        if (tst.written.find(name) == tst.written.end()) {
            fprintf(stderr, "dump_assistant: the target's %s never appeared in the graph\n", name.c_str());
            return 1;
        }
    }
    st.written.insert(tst.written.begin(), tst.written.end());
    std::memcpy(h.data(), llama_get_embeddings_nextn_ith(ctx_tgt, n - 1), h.size() * sizeof(float));
    llama_token id = argmax(llama_get_logits_ith(ctx_tgt, -1), n_vocab);

    // The input the Go side has to reproduce before any of the rest means
    // anything: the target's normed state at the last prompt position.
    write_floats(out_dir + "/target_h.bin", h.data(), h.size());
    st.written["target_h"] = dumped{ "target_h.bin", { n_embd_out, 1, 1, 1 } };

    // The recorded draft.
    const llama_token first_guess = guess(id, n);
    st.active = false;
    for (const auto & name : st.wanted) {
        if (st.written.find(name) != st.written.end()) continue;
        if (optional.find(name) != optional.end()) continue;
        fprintf(stderr, "dump_assistant: %s never appeared in the graph\n", name.c_str());
        return 1;
    }

    std::vector<std::pair<int, float>> top;
    {
        const float * logits = llama_get_logits_ith(ctx_dft, 0);
        top.reserve(n_vocab);
        for (int i = 0; i < n_vocab; ++i) top.emplace_back(i, logits[i]);
        std::partial_sort(top.begin(), top.begin() + 64, top.end(),
                          [](const auto & a, const auto & b) { return a.second > b.second; });
        top.resize(64);
    }

    // The greedy run. Each step drafts before the target reads the token, so
    // the assistant sees the target's cache as the speculation would: every
    // position before the one it drafts from.
    std::vector<llama_token> greedy, drafts, drafts2;
    int pos = n;
    for (int step = 0; step < run.steps; ++step) {
        drafts.push_back(step == 0 ? first_guess : guess(id, pos));
        drafts2.push_back(second(drafts.back(), pos));
        greedy.push_back(id);
        if (llama_decode(ctx_tgt, llama_batch_get_one(&id, 1))) die("the target's decode failed");
        std::memcpy(h.data(), llama_get_embeddings_nextn_ith(ctx_tgt, 0), h.size() * sizeof(float));
        id = argmax(llama_get_logits_ith(ctx_tgt, -1), n_vocab);
        ++pos;
    }

    const std::string path = out_dir + "/index.json";
    FILE * f = fopen(path.c_str(), "w");
    if (!f) die(("cannot write " + path).c_str());
    fprintf(f, "{\n");
    fprintf(f, "  \"model\": \"%s\",\n", base_name(target_path).c_str());
    fprintf(f, "  \"assistant\": \"%s\",\n", base_name(assistant_path).c_str());
    fprintf(f, "  \"prompt\": \"%s\",\n", json_escape(run.label).c_str());
    fprintf(f, "  \"n_embd\": %d,\n  \"n_layer\": %d,\n  \"n_embd_per_layer\": 0,\n",
            llama_model_n_embd(assistant), n_layer);
    write_ids(f, "tokens", tokens, false);
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
    fprintf(f, "  \"argmax\": %d,\n", top[0].first);
    write_ids(f, "greedy", greedy, false);
    write_ids(f, "drafts", drafts, false);
    write_ids(f, "drafts2", drafts2, true);
    fprintf(f, "}\n");
    fclose(f);

    free(draft.token);
    draft.token = nullptr;
    llama_batch_free(draft);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(assistant);
    llama_model_free(target);
    llama_backend_free();
    fprintf(stderr, "dump_assistant: wrote %zu tensors to %s\n", st.written.size(), out_dir.c_str());
    return 0;
}
