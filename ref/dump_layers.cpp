// Records the intermediate activations of a forward pass, as llama.cpp
// computes them, so that a Go engine can be checked against them without
// llama.cpp being present at test time.
//
// llama.cpp names every waypoint of its graph ("attn_norm-0", "l_out-34",
// "result_norm"), and it uses the same names whatever the architecture. A
// scheduler callback intercepts them by name and writes the float32 contents
// out. So nothing here is specific to one model, and nothing here should be:
// which waypoints to keep, on which prompt, is described by a run file the
// caller names, and lives under ref/<model>/ beside the engine that wants it.
//
// One directive per line, '#' starts a comment, a blank line is ignored:
//
//   prompt        <text>        the prompt; \n is a newline, \s a space, \\ a
//                               backslash. A leading or trailing space must be
//                               written \s, because the line is trimmed.
//   repeat        <n>           repeat the prompt n times (default 1)
//   label         <text>        what index.json records as "prompt"; defaults
//                               to the prompt itself
//   add_special   <0|1>         llama_tokenize's add_special (default 1)
//   min_tokens    <n>           fail if the run tokenizes to fewer (default 0)
//   blocks        <i> <i> ...   the block indices the per-block names apply to
//   all_blocks    <name>        a name recorded for every block, 0..n_layer-1
//   require       <name>        a per-block name that must appear
//   optional      <name>        a per-block name that may be absent
//   global        <name>        a whole-model name that must appear
//   global_opt    <name>        a whole-model name that may be absent
//   last_column   <0|1>         keep only the last column of each recording
//
// A per-block name is expanded to "<name>-<index>" for each block listed. A
// name that is required and never appears in the graph is an error: it means
// the run file and llama.cpp disagree about what this architecture computes,
// which is worth stopping for rather than recording a hole.

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

// The run file. Everything model-specific that used to live in this source
// now arrives through one of these.
struct run_spec {
    std::string prompt;
    std::string label;
    int  repeat      = 1;
    bool add_special = true;
    int  min_tokens  = 0;
    bool last_column = false;
    std::vector<int>         blocks;
    std::vector<std::string> all_blocks;
    std::vector<std::string> require_names;
    std::vector<std::string> optional_names;
    std::vector<std::string> global_names;
    std::vector<std::string> global_opt;
};

static std::string unescape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 == s.size()) { out += s[i]; continue; }
        switch (s[++i]) {
            case 'n':    out += '\n'; break;
            case 's':    out += ' ';  break;
            case '\\': out += '\\'; break;
            default:     die("unknown escape in a run file");
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
        // Trailing whitespace goes, which is why a significant trailing space
        // has to be written \s.
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
        if (s.empty() || s[0] == '#') continue;

        const size_t sp  = s.find(' ');
        const std::string key = s.substr(0, sp);
        std::string val = sp == std::string::npos ? "" : s.substr(sp + 1);
        // A directive and its value are separated by whitespace; the run files
        // align their values into columns, and that alignment is not part of
        // the value. A leading space that is part of a prompt is written \s,
        // which survives this because it is not yet a space.
        const size_t first = val.find_first_not_of(' ');
        val = first == std::string::npos ? "" : val.substr(first);

        if      (key == "prompt")      r.prompt = unescape(val);
        else if (key == "label")     { r.label  = unescape(val); have_label = true; }
        else if (key == "repeat")      r.repeat = atoi(val.c_str());
        else if (key == "add_special") r.add_special = atoi(val.c_str()) != 0;
        else if (key == "min_tokens")  r.min_tokens  = atoi(val.c_str());
        else if (key == "last_column") r.last_column = atoi(val.c_str()) != 0;
        else if (key == "require")     r.require_names.push_back(val);
        else if (key == "optional")    r.optional_names.push_back(val);
        else if (key == "global")      r.global_names.push_back(val);
        else if (key == "global_opt")  r.global_opt.push_back(val);
        else if (key == "all_blocks")  r.all_blocks.push_back(val);
        else if (key == "blocks") {
            const char * p   = val.c_str();
            char       * end = nullptr;
            for (long v = strtol(p, &end, 10); end != p; v = strtol(p, &end, 10)) {
                r.blocks.push_back((int) v);
                p = end;
            }
        } else die(("unknown directive: " + key).c_str());
    }
    fclose(f);

    if (r.prompt.empty()) die("the run file declares no prompt");
    if (!have_label) r.label = r.prompt;
    return r;
}

// The wanted set, and the subset of it that is allowed to be absent. This is
// given the real block count, so it never asks for a block that does not
// exist — which is why nothing has to prune the set afterwards.
static void expand_names(const run_spec & r, int n_layer,
                         std::set<std::string> & wanted,
                         std::set<std::string> & optional) {
    for (const auto & n : r.global_names) wanted.insert(n);
    for (const auto & n : r.global_opt) { wanted.insert(n); optional.insert(n); }

    for (int il : r.blocks) {
        if (il >= n_layer) continue;
        const std::string s = "-" + std::to_string(il);
        for (const auto & n : r.require_names) wanted.insert(n + s);
        for (const auto & n : r.optional_names) {
            wanted.insert(n + s);
            optional.insert(n + s);
        }
    }
    for (const auto & n : r.all_blocks) {
        for (int il = 0; il < n_layer; ++il) wanted.insert(n + "-" + std::to_string(il));
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
        fprintf(stderr, "usage: dump_layers <model.gguf> <out-dir> <run-file>\n");
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string out_dir    = argv[2];
    const run_spec    run        = read_run(argv[3]);

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

    std::string prompt;
    for (int i = 0; i < run.repeat; ++i) prompt += run.prompt;

    std::vector<llama_token> tokens(8192);
    int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                           tokens.data(), (int32_t) tokens.size(),
                           run.add_special, false);
    if (n <= 0) die("tokenization failed");
    tokens.resize(n);
    if (n < run.min_tokens) die("the run tokenized to fewer than min_tokens");
    fprintf(stderr, "dump_layers: %d tokens\n", n);

    dump_state st;
    st.dir = out_dir;
    st.last_column_only = run.last_column;
    std::set<std::string> optional;
    expand_names(run, llama_model_n_layer(model), st.wanted, optional);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = 8192;
    cparams.n_batch           = 8192;
    cparams.n_ubatch          = 8192;   // one graph, so each name fires once
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
    write_index(st, run.label,
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
