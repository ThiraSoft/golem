// dump_audio — what llama.cpp computes for one piece of audio.
//
// The fifth recorder, and dump_vision with a sound file in place of a picture.
// It links libmtmd for the same reason: the conformer's graph is built by
// clip.cpp and llama_decode never sees it.
//
// One thing makes it different. clip_graph_gemma4a names almost none of its
// nodes, so a waypoint cannot be asked for by name the way a vision one is.
// It is asked for by position instead: `--list` runs the graph and prints one
// line per node the evaluation callback is handed — its index, its operation,
// its name if it has one, and its four dimensions — and the run file then says
// `node 47 pre_encode`. The indices are the callback's own sequence rather
// than a walk of the built graph, so the two can never disagree.
//
// usage: dump_audio [--list] <model.gguf> <mmproj.gguf> <out-dir> <run-file>
//        with --list, <out-dir> is ignored and nothing is written.

#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

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

// The tower's state: waypoints are indices into the callback's sequence, and
// each carries the label the recording files it under.
struct tower_state {
    std::string dir;
    std::map<int, std::string> wanted;   // node index -> label
    std::map<std::string, dumped> written;
    int  seen   = 0;      // how many nodes the callback has been offered
    bool list   = false;
    bool active = true;
};

// The text model's state: those nodes do have names, so this half is
// dump_vision's callback unchanged.
struct text_state {
    std::string dir;
    std::set<std::string> wanted;
    std::map<std::string, dumped> written;
    bool active = true;
};

static void die(const char * what) {
    fprintf(stderr, "dump_audio: %s\n", what);
    exit(1);
}

static void write_floats(const std::string & path, const std::vector<float> & v) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) die(("cannot write " + path).c_str());
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}

static void save(const std::string & dir, const std::string & label,
                 struct ggml_tensor * t, std::map<std::string, dumped> & into) {
    if (t->type != GGML_TYPE_F32) die((label + " is not F32").c_str());
    if (!ggml_is_contiguous(t))   die((label + " is not contiguous").c_str());
    std::vector<float> all(ggml_nelements(t));
    ggml_backend_tensor_get(t, all.data(), 0, ggml_nbytes(t));
    dumped d;
    d.file = label + ".bin";
    write_floats(dir + "/" + d.file, all);
    for (int i = 0; i < 4; ++i) d.ne[i] = t->ne[i];
    into[label] = d;
}

static bool on_tower_node(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (tower_state *) user_data;
    if (!st->active) return false;

    if (ask) {
        const int index = st->seen++;
        if (st->list) {
            fprintf(stdout, "%4d  %-12s %-24s [%lld, %lld, %lld, %lld]\n",
                    index, ggml_op_name(t->op), ggml_get_name(t),
                    (long long) t->ne[0], (long long) t->ne[1],
                    (long long) t->ne[2], (long long) t->ne[3]);
            return false;
        }
        return st->wanted.find(index) != st->wanted.end();
    }
    // The index of the node being handed back is the one just counted.
    auto it = st->wanted.find(st->seen - 1);
    if (it == st->wanted.end()) return true;
    save(st->dir, it->second, t, st->written);
    return true;
}

static bool on_text_node(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (text_state *) user_data;
    if (!st->active) return false;
    const std::string name = ggml_get_name(t);
    if (st->wanted.find(name) == st->wanted.end()) return false;
    if (ask) return true;
    save(st->dir, name, t, st->written);
    return true;
}

struct run_spec {
    std::string audio;
    std::string prompt;
    int  n_predict = 16;
    std::vector<std::pair<int, std::string>> nodes;   // index, label
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
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
        if (s.empty() || s[0] == '#') continue;

        const size_t sp = s.find(' ');
        const std::string key = s.substr(0, sp);
        std::string val = sp == std::string::npos ? "" : s.substr(sp + 1);
        const size_t first = val.find_first_not_of(' ');
        val = first == std::string::npos ? "" : val.substr(first);

        if      (key == "audio")      r.audio  = val;
        else if (key == "prompt")     r.prompt = unescape(val);
        else if (key == "n_predict")  r.n_predict = atoi(val.c_str());
        else if (key == "global")     r.global_names.push_back(val);
        else if (key == "global_opt") r.global_opt.push_back(val);
        else if (key == "node") {
            const size_t at = val.find(' ');
            if (at == std::string::npos) die("a node line is an index and a label");
            std::string label = val.substr(at + 1);
            const size_t l = label.find_first_not_of(' ');
            label = l == std::string::npos ? "" : label.substr(l);
            if (label.empty()) die("a node line is an index and a label");
            r.nodes.emplace_back(atoi(val.substr(0, at).c_str()), label);
        }
        else die(("unknown directive: " + key).c_str());
    }
    fclose(f);

    if (r.audio.empty())  die("the run file names no audio");
    if (r.prompt.empty()) die("the run file declares no prompt");
    return r;
}

int main(int argc, char ** argv) {
    int arg = 1;
    bool list = false;
    if (argc > 1 && strcmp(argv[1], "--list") == 0) { list = true; arg = 2; }
    if (argc - arg != 4) {
        fprintf(stderr, "usage: dump_audio [--list] <model.gguf> <mmproj.gguf> <out-dir> <run-file>\n");
        return 1;
    }
    const std::string model_path  = argv[arg + 0];
    const std::string mmproj_path = argv[arg + 1];
    const std::string out_dir     = argv[arg + 2];
    const run_spec    run         = read_run(argv[arg + 3]);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    static ggml_backend_dev_t no_devices[] = { nullptr };
    mparams.devices = no_devices;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) die("cannot load the model");
    const llama_vocab * vocab = llama_model_get_vocab(model);

    tower_state tower;
    tower.dir  = out_dir;
    tower.list = list;
    for (const auto & n : run.nodes) tower.wanted[n.first] = n.second;

    text_state text;
    text.dir = out_dir;
    std::set<std::string> optional;
    for (const auto & n : run.global_names) text.wanted.insert(n);
    for (const auto & n : run.global_opt) { text.wanted.insert(n); optional.insert(n); }
    if (list) text.active = false;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 8192;
    cparams.n_batch         = 8192;
    cparams.n_ubatch        = 8192;
    cparams.n_threads       = 8;
    cparams.n_threads_batch = 8;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval           = on_text_node;
    cparams.cb_eval_user_data = &text;
    llama_context * lctx = llama_init_from_model(model, cparams);
    if (!lctx) die("cannot create the context");

    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu            = false;
    vparams.print_timings      = false;
    vparams.n_threads          = 8;
    vparams.warmup             = false;  // a warmup pass would fire the callback first
    vparams.flash_attn_type    = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    vparams.cb_eval            = on_tower_node;
    vparams.cb_eval_user_data  = &tower;
    mtmd_context * vctx = mtmd_init_from_file(mmproj_path.c_str(), model, vparams);
    if (!vctx) die("cannot load the projector");
    if (!mtmd_support_audio(vctx)) die("the projector carries no audio encoder");

    auto wrapper = mtmd_helper_bitmap_init_from_file(vctx, run.audio.c_str(), false);
    if (!wrapper.bitmap) die("cannot read the audio");
    const mtmd_bitmap * bitmaps[1] = { wrapper.bitmap };

    const std::string prompt = std::string("<|turn>user\n") + mtmd_default_marker() +
                               "\n" + run.prompt + "<turn|>\n<|turn>model\n";
    mtmd_input_text mtext;
    mtext.text          = prompt.c_str();
    mtext.add_special   = true;
    mtext.parse_special = true;

    // Timed, because the front end is on this side of the line: mtmd computes
    // the mel here and the "slice encoded in" line further down covers only
    // the graph. A comparison against an engine that does both in one call
    // needs the two numbers.
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const int64_t t_tok = ggml_time_ms();
    if (mtmd_tokenize(vctx, chunks, &mtext, bitmaps, 1)) die("tokenization failed");
    fprintf(stderr, "dump_audio: preprocessed and tokenized in %lld ms\n",
            (long long) (ggml_time_ms() - t_tok));

    std::vector<llama_token> tokens;
    int audio_start = -1, audio_len = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * t = mtmd_input_chunk_get_tokens_text(c, &n);
            tokens.insert(tokens.end(), t, t + n);
        } else {
            audio_start = (int) tokens.size();
            audio_len   = (int) mtmd_input_chunk_get_n_tokens(c);
            tokens.insert(tokens.end(), (size_t) audio_len, (llama_token) 0);
        }
    }
    if (audio_start < 0) die("the prompt produced no audio chunk");
    fprintf(stderr, "dump_audio: %zu tokens, %d of them one recording at %d\n",
            tokens.size(), audio_len, audio_start);

    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(vctx, lctx, chunks, 0, 0, 8192, true, &n_past))
        die("evaluating the chunks failed");

    if (list) {
        fprintf(stderr, "dump_audio: %d nodes went through the tower's graph\n", tower.seen);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(wrapper.bitmap);
        mtmd_free(vctx);
        llama_free(lctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    tower.active = false;
    text.active  = false;   // the greedy continuation below must not overwrite

    for (const auto & n : run.nodes) {
        if (tower.written.find(n.second) == tower.written.end()) {
            fprintf(stderr, "dump_audio: node %d (%s) never appeared; "
                            "run --list again, the graph moved\n", n.first, n.second.c_str());
            return 1;
        }
    }
    for (const auto & name : text.wanted) {
        if (text.written.find(name) != text.written.end()) continue;
        if (optional.find(name) != optional.end()) continue;
        fprintf(stderr, "dump_audio: %s never appeared in the graph\n", name.c_str());
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(lctx, -1);
    int argmax = 0;
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[argmax]) argmax = i;

    std::vector<llama_token> greedy;
    llama_token next = argmax;
    for (int step = 0; step < run.n_predict; ++step) {
        greedy.push_back(next);
        if (llama_decode(lctx, llama_batch_get_one(&next, 1))) die("decode failed");
        const float * l = llama_get_logits_ith(lctx, -1);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (l[i] > l[best]) best = i;
        next = best;
    }

    const std::string path = out_dir + "/index.json";
    FILE * f = fopen(path.c_str(), "w");
    if (!f) die(("cannot write " + path).c_str());
    fprintf(f, "{\n");
    fprintf(f, "  \"model\": \"%s\",\n", model_path.c_str());
    fprintf(f, "  \"prompt\": \"%s\",\n", run.prompt.c_str());
    fprintf(f, "  \"audio\": \"%s\",\n", run.audio.c_str());
    fprintf(f, "  \"n_embd\": %d,\n", llama_model_n_embd(model));
    fprintf(f, "  \"audio_start\": %d,\n  \"n_audio_tokens\": %d,\n", audio_start, audio_len);
    fprintf(f, "  \"tokens\": [");
    for (size_t i = 0; i < tokens.size(); ++i) fprintf(f, "%s%d", i ? ", " : "", tokens[i]);
    fprintf(f, "],\n");
    fprintf(f, "  \"tensors\": {\n");
    std::map<std::string, dumped> all = tower.written;
    all.insert(text.written.begin(), text.written.end());
    size_t i = 0;
    for (const auto & kv : all) {
        fprintf(f, "    \"%s\": {\"file\": \"%s\", \"ne\": [%lld, %lld, %lld, %lld]}%s\n",
                kv.first.c_str(), kv.second.file.c_str(),
                (long long) kv.second.ne[0], (long long) kv.second.ne[1],
                (long long) kv.second.ne[2], (long long) kv.second.ne[3],
                ++i == all.size() ? "" : ",");
    }
    fprintf(f, "  },\n");
    fprintf(f, "  \"argmax\": %d,\n", argmax);
    fprintf(f, "  \"greedy\": [");
    for (size_t j = 0; j < greedy.size(); ++j) fprintf(f, "%s%d", j ? ", " : "", greedy[j]);
    fprintf(f, "]\n}\n");
    fclose(f);

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(wrapper.bitmap);
    mtmd_free(vctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "dump_audio: wrote %zu tensors to %s\n", all.size(), out_dir.c_str());
    return 0;
}
