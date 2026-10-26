// Records the log-probability llama.cpp gives each option of a closed
// question, token by token, so that golem's /v1/systemone can be held to it.
//
// A case is a prompt already rendered by a chat template and the options that
// may follow it. Each option is read as a continuation: the prompt and the
// option are tokenized together, the prompt's tokens have to be a prefix of
// that, and what is past them is the option. The whole sequence is decoded
// from an empty cache, every position past the prompt's last but one asking
// for its logits, and each option token's log-softmax is written down. The
// sums are what golem compares; the tokens are written too, so the Go side
// can be fed exactly what llama.cpp read.
//
// The corpus is tab-separated, with dump_tokens' escapes, an option a column:
//
//   name <TAB> prompt <TAB> option <TAB> option …
//
// It runs on the processor (no layer offloaded, no product handed to a card),
// in llama.cpp's own order of arithmetic, which is what the other Gemma
// fixtures are recorded under.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void die(const char * what) {
    fprintf(stderr, "dump_logprobs: %s\n", what);
    exit(1);
}

static std::string escape(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out + "\"";
}

static std::string unescape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 == s.size()) { out += s[i]; continue; }
        switch (s[++i]) {
            case 'n':  out += '\n'; break;
            case 't':  out += '\t'; break;
            case 'r':  out += '\r'; break;
            case 's':  out += ' ';  break;
            case '\\': out += '\\'; break;
            default: die("unknown escape in the corpus");
        }
    }
    return out;
}

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == sep) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

struct testcase {
    std::string name, prompt;
    std::vector<std::string> options;
};

static std::vector<testcase> read_corpus(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) die(("cannot read " + path).c_str());
    std::vector<testcase> cases;
    std::string s;
    int c;
    auto flush = [&]() {
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (!s.empty() && s[0] != '#') {
            std::vector<std::string> col = split(s, '\t');
            if (col.size() < 3) die("a corpus line has no option");
            testcase tc{col[0], unescape(col[1]), {}};
            for (size_t k = 2; k < col.size(); k++) tc.options.push_back(unescape(col[k]));
            cases.push_back(tc);
        }
        s.clear();
    };
    while ((c = fgetc(f)) != EOF) {
        s += (char) c;
        if (c == '\n') flush();
    }
    flush();
    fclose(f);
    if (cases.empty()) die("the corpus is empty");
    return cases;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> ids(text.size() + 16);
    const int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(),
                                 ids.data(), (int32_t) ids.size(), false, true);
    if (n < 0) die("the token buffer was too small");
    ids.resize(n);
    return ids;
}

static void write_ids(FILE * f, const std::vector<llama_token> & ids, size_t from) {
    fprintf(f, "[");
    for (size_t j = from; j < ids.size(); j++) fprintf(f, "%s%d", j > from ? ", " : "", ids[j]);
    fprintf(f, "]");
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: dump_logprobs <model.gguf> <out-dir> <cases.tsv>\n");
        return 1;
    }
    const std::vector<testcase> cases = read_corpus(argv[3]);
    const std::string out_dir = argv[2];

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) die("cannot load the model");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 4096;
    cparams.n_ubatch = 512;
    // A build with a card would otherwise hand the prompt's products to it
    // while the weights stay in host memory, and the record would be neither
    // device's arithmetic.
    cparams.op_offload = false;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) die("cannot make a context");

    FILE * f = fopen((out_dir + "/logprobs.json").c_str(), "wb");
    if (!f) die("cannot write logprobs.json");
    fprintf(f, "{\n  \"cases\": [\n");
    for (size_t ci = 0; ci < cases.size(); ci++) {
        const testcase & tc = cases[ci];
        const std::vector<llama_token> prompt = tokenize(vocab, tc.prompt);
        fprintf(f, "    {\"name\": %s, \"prompt\": ", escape(tc.name).c_str());
        write_ids(f, prompt, 0);
        fprintf(f, ", \"options\": [\n");
        for (size_t oi = 0; oi < tc.options.size(); oi++) {
            const std::vector<llama_token> ids = tokenize(vocab, tc.prompt + tc.options[oi]);
            if (ids.size() <= prompt.size()) die("an option added no token");
            for (size_t j = 0; j < prompt.size(); j++) {
                if (ids[j] != prompt[j]) die("an option changed the prompt's tokens");
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
            for (size_t j = 0; j < ids.size(); j++) {
                batch.token[j] = ids[j];
                batch.pos[j] = (llama_pos) j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = j + 1 >= prompt.size() && j + 1 < ids.size();
            }
            batch.n_tokens = (int32_t) ids.size();
            if (llama_decode(ctx, batch) != 0) die("llama_decode failed");

            fprintf(f, "      {\"text\": %s, \"tokens\": ", escape(tc.options[oi]).c_str());
            write_ids(f, ids, prompt.size());
            fprintf(f, ", \"logprobs\": [");
            double sum = 0;
            for (size_t j = prompt.size(); j < ids.size(); j++) {
                const float * row = llama_get_logits_ith(ctx, (int32_t) (j - 1));
                double m = -INFINITY;
                for (int v = 0; v < n_vocab; v++) m = fmax(m, row[v]);
                double z = 0;
                for (int v = 0; v < n_vocab; v++) z += exp((double) row[v] - m);
                const double lp = row[ids[j]] - m - log(z);
                sum += lp;
                fprintf(f, "%s%.9g", j > prompt.size() ? ", " : "", lp);
            }
            fprintf(f, "], \"sum\": %.9g}%s\n", sum, oi + 1 == tc.options.size() ? "" : ",");
            llama_batch_free(batch);
        }
        fprintf(f, "    ]}%s\n", ci + 1 == cases.size() ? "" : ",");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf("dump_logprobs: wrote %zu cases to %s/logprobs.json\n", cases.size(), out_dir.c_str());
    return 0;
}
