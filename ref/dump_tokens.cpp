// Records what llama.cpp makes of a corpus of sentences, so a Go tokenizer can
// be checked against it without llama.cpp being present at test time.
//
// Only the vocabulary is loaded — vocab_only skips the gigabytes of weights —
// and each case is written with the flags it was tokenized under, because
// add_special and parse_special change the answer.
//
// The corpus is a file rather than an array in this source, because the cases
// that matter are the ones a given tokenizer gets wrong, and those differ by
// model: Gemma's special tokens are not Qwen's, and Gemma has no byte alphabet
// to get wrong. Four tab-separated columns:
//
//   name <TAB> text <TAB> add_special <TAB> parse_special
//
// The text column uses \n, \t, \r, \s, \\ and \xNN, because a tab-separated
// file cannot carry a literal tab, a line cannot carry a literal newline, and
// a trailing space would not survive being read. A line starting with # is a
// comment; a blank line is ignored.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct testcase {
    std::string name;
    std::string text;
    bool add_special;
    bool parse_special;
};

static void die(const char * what) {
    fprintf(stderr, "dump_tokens: %s\n", what);
    exit(1);
}

// JSON string escaping, enough for what the corpus contains.
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

// The corpus file's escapes. Distinct from JSON's: this one has to survive a
// tab-separated line, so a tab and a newline cannot be written literally, and
// a significant trailing space has to be written \s.
static std::string unescape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 == s.size()) { out += s[i]; continue; }
        switch (s[++i]) {
            case 'n':    out += '\n'; break;
            case 't':    out += '\t'; break;
            case 'r':    out += '\r'; break;
            case 's':    out += ' ';  break;
            case '\\': out += '\\'; break;
            case 'x': {
                if (i + 2 >= s.size()) die("truncated \\x escape in the corpus");
                out += (char) strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
                i += 2;
                break;
            }
            default: die("unknown escape in the corpus");
        }
    }
    return out;
}

static std::vector<testcase> read_corpus(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) die(("cannot read " + path).c_str());

    std::vector<testcase> cases;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        std::string s(line);
        // Only the line ending goes: a trailing tab separates an empty column
        // from a present one, and a trailing space is written \s anyway.
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (s.empty() || s[0] == '#') continue;

        std::vector<std::string> col;
        size_t start = 0;
        for (size_t i = 0; i <= s.size(); ++i) {
            if (i == s.size() || s[i] == '\t') {
                col.push_back(s.substr(start, i - start));
                start = i + 1;
            }
        }
        if (col.size() != 4) {
            die(("a corpus line has " + std::to_string(col.size()) + " columns, not 4").c_str());
        }
        cases.push_back({col[0], unescape(col[1]), col[2] == "1", col[3] == "1"});
    }
    fclose(f);
    if (cases.empty()) die("the corpus is empty");
    return cases;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: dump_tokens <model.gguf> <out-dir> <corpus.tsv>\n");
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string out_dir    = argv[2];
    const std::vector<testcase> cases = read_corpus(argv[3]);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) die("cannot load the model");

    const llama_vocab * vocab = llama_model_get_vocab(model);

    FILE * f = fopen((out_dir + "/cases.json").c_str(), "wb");
    if (!f) die("cannot write cases.json");

    fprintf(f, "{\n  \"cases\": [\n");
    const int n_cases = (int) cases.size();
    for (int i = 0; i < n_cases; i++) {
        const testcase & c = cases[i];
        const std::string text = c.text;

        std::vector<llama_token> ids(text.size() * 4 + 16);
        const int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(),
                                     ids.data(), (int32_t) ids.size(),
                                     c.add_special, c.parse_special);
        if (n < 0) die("the token buffer was too small");
        ids.resize(n);

        // Detokenized with specials rendered and nothing removed, so the Go
        // side is compared against the whole of what llama.cpp writes back.
        std::vector<char> buf(text.size() * 8 + 64);
        const int m = llama_detokenize(vocab, ids.data(), n,
                                       buf.data(), (int32_t) buf.size(),
                                       false, true);
        if (m < 0) die("the text buffer was too small");
        const std::string back(buf.data(), m);

        fprintf(f, "    {\"name\": %s, \"text\": %s, \"add_special\": %s, \"parse_special\": %s, \"ids\": [",
                escape(c.name).c_str(), escape(text).c_str(),
                c.add_special ? "true" : "false",
                c.parse_special ? "true" : "false");
        for (int j = 0; j < n; j++) {
            fprintf(f, "%s%d", j ? ", " : "", ids[j]);
        }
        fprintf(f, "], \"detokenized\": %s}%s\n",
                escape(back).c_str(), i + 1 == n_cases ? "" : ",");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);

    llama_model_free(model);
    llama_backend_free();

    printf("dump_tokens: wrote %d cases to %s/cases.json\n", n_cases, out_dir.c_str());
    return 0;
}
