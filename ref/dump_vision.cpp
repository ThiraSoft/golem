// dump_vision — what llama.cpp computes for one image.
//
// The fourth recorder, and the only one that does not go in through the llama
// API. A vision tower is built by clip.cpp, whose graph llama_decode never
// sees: the way in is mtmd, which tokenizes a prompt carrying a media marker,
// encodes the picture, and hands the embeddings to the text model. So this
// tool links libmtmd and includes its internal-ish headers, where the other
// three link llama alone.
//
// It records three things:
//   - the waypoints of the vision graph, by the names models/gemma4v.cpp gives
//     its nodes, as raw F32 in ggml's layout;
//   - the token layout the prompt became: the text tokens, and where the image
//     chunk sits among them;
//   - the greedy continuation, which is the end-to-end check.
//
// The run file names the image, the prompt and the waypoints. Its grammar is
// dump_layers' own, minus what does not apply and plus `image`.
//
// usage: dump_vision <model.gguf> <mmproj.gguf> <out-dir> <run-file>

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

struct dump_state {
    std::string           dir;
    std::set<std::string> wanted;
    std::map<std::string, dumped> written;
    bool                  active = true;
};

static void die(const char * what) {
    fprintf(stderr, "dump_vision: %s\n", what);
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

    std::vector<float> all(ggml_nelements(t));
    ggml_backend_tensor_get(t, all.data(), 0, ggml_nbytes(t));

    dumped d;
    d.file = name + ".bin";
    write_floats(st->dir + "/" + d.file, all);
    for (int i = 0; i < 4; ++i) d.ne[i] = t->ne[i];
    st->written[name] = d;
    return true;
}

struct run_spec {
    std::string image;
    std::string prompt;
    int  n_predict = 16;
    bool all_blocks = false;
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

        if      (key == "image")      r.image  = val;
        else if (key == "prompt")     r.prompt = unescape(val);
        else if (key == "n_predict")  r.n_predict = atoi(val.c_str());
        else if (key == "require")    r.require_names.push_back(val);
        else if (key == "global")     r.global_names.push_back(val);
        else if (key == "global_opt") r.global_opt.push_back(val);
        else if (key == "blocks")     r.all_blocks = (val == "all");
        else die(("unknown directive: " + key).c_str());
    }
    fclose(f);

    if (r.image.empty())  die("the run file names no image");
    if (r.prompt.empty()) die("the run file declares no prompt");
    return r;
}

static void expand_names(const run_spec & r, int n_block,
                         std::set<std::string> & wanted,
                         std::set<std::string> & optional) {
    for (const auto & n : r.global_names) wanted.insert(n);
    for (const auto & n : r.global_opt) { wanted.insert(n); optional.insert(n); }
    if (!r.all_blocks) return;
    for (int il = 0; il < n_block; ++il) {
        const std::string s = "-" + std::to_string(il);
        for (const auto & n : r.require_names) wanted.insert(n + s);
    }
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: dump_vision <model.gguf> <mmproj.gguf> <out-dir> <run-file>\n");
        return 1;
    }
    const std::string model_path  = argv[1];
    const std::string mmproj_path = argv[2];
    const std::string out_dir     = argv[3];
    const run_spec    run         = read_run(argv[4]);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    // An empty device list keeps every graph on the CPU, the tower's included.
    static ggml_backend_dev_t no_devices[] = { nullptr };
    mparams.devices = no_devices;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) die("cannot load the model");
    const llama_vocab * vocab = llama_model_get_vocab(model);

    dump_state st;
    st.dir = out_dir;
    std::set<std::string> optional;
    // The tower's block count is not the language model's. It is read back
    // from the graph instead of being declared: a name that never fires is
    // reported below, and one block too many is harmless.
    expand_names(run, 64, st.wanted, optional);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 8192;
    cparams.n_batch         = 8192;
    cparams.n_ubatch        = 8192;
    cparams.n_threads       = 8;
    cparams.n_threads_batch = 8;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * lctx = llama_init_from_model(model, cparams);
    if (!lctx) die("cannot create the context");

    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu            = false;
    vparams.print_timings      = false;
    vparams.n_threads          = 8;
    vparams.warmup             = false;  // a warmup pass would fire the callback first
    vparams.flash_attn_type    = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    vparams.cb_eval            = on_node;
    vparams.cb_eval_user_data  = &st;
    mtmd_context * vctx = mtmd_init_from_file(mmproj_path.c_str(), model, vparams);
    if (!vctx) die("cannot load the projector");
    if (!mtmd_support_vision(vctx)) die("the projector carries no vision encoder");

    auto wrapper = mtmd_helper_bitmap_init_from_file(vctx, run.image.c_str(), false);
    if (!wrapper.bitmap) die("cannot read the image");
    const mtmd_bitmap * bitmaps[1] = { wrapper.bitmap };

    // The marker is where the picture goes. The prompt around it is written in
    // the checkpoint's own turn markers, so what is recorded is a real
    // conversation rather than a bare caption.
    const std::string prompt = std::string("<|turn>user\n") + mtmd_default_marker() +
                               "\n" + run.prompt + "<turn|>\n<|turn>model\n";
    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.add_special   = true;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize(vctx, chunks, &text, bitmaps, 1)) die("tokenization failed");

    // The layout, before anything is run: the text tokens in order, with the
    // image chunk's extent noted. The Go side rebuilds exactly this.
    std::vector<llama_token> tokens;
    int image_start = -1, image_len = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * t = mtmd_input_chunk_get_tokens_text(c, &n);
            tokens.insert(tokens.end(), t, t + n);
        } else {
            image_start = (int) tokens.size();
            image_len   = (int) mtmd_input_chunk_get_n_tokens(c);
            // The soft tokens have no identifiers. They are held here by the
            // padding token, which is also whose per-layer input llama.cpp
            // uses for an embedding batch.
            tokens.insert(tokens.end(), (size_t) image_len, (llama_token) 0);
        }
    }
    if (image_start < 0) die("the prompt produced no image chunk");
    fprintf(stderr, "dump_vision: %zu tokens, %d of them one picture at %d\n",
            tokens.size(), image_len, image_start);

    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(vctx, lctx, chunks, 0, 0, 8192, true, &n_past))
        die("evaluating the chunks failed");

    st.active = false;   // the greedy continuation below must not overwrite

    for (const auto & name : st.wanted) {
        if (st.written.find(name) != st.written.end()) continue;
        if (optional.find(name) != optional.end()) continue;
        // A block the tower does not have is not an error: the set is padded.
        if (name.find("-") != std::string::npos) continue;
        fprintf(stderr, "dump_vision: %s never appeared in the graph\n", name.c_str());
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(lctx, -1);
    std::vector<std::pair<int, float>> top;
    top.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) top.emplace_back(i, logits[i]);
    std::partial_sort(top.begin(), top.begin() + 64, top.end(),
                      [](const auto & a, const auto & b) { return a.second > b.second; });
    top.resize(64);

    std::vector<llama_token> greedy;
    llama_token next = top[0].first;
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
    fprintf(f, "  \"prompt\": \"%s\",\n", run.prompt.c_str());
    fprintf(f, "  \"image\": \"%s\",\n", run.image.c_str());
    fprintf(f, "  \"n_embd\": %d,\n", llama_model_n_embd(model));
    fprintf(f, "  \"image_start\": %d,\n  \"n_image_tokens\": %d,\n", image_start, image_len);
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
    fprintf(f, "  \"argmax\": %d,\n", top.empty() ? -1 : top[0].first);
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
    fprintf(stderr, "dump_vision: wrote %zu tensors to %s\n", st.written.size(), out_dir.c_str());
    return 0;
}
