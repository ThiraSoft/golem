// Records what llama.cpp makes of a corpus of sentences, so the Go tokenizer
// can be checked against it without llama.cpp being present at test time.
//
// Only the vocabulary is loaded — vocab_only skips the 3.2 GB of weights — and
// each case is written with the flags it was tokenized under, because
// add_special and parse_special change the answer.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct testcase {
    const char * name;
    const char * text;
    bool add_special;
    bool parse_special;
};

// The corpus exists to make the branches of the tokenizer differ from one
// another: escaping, runs of newlines, byte fallback, and the special tokens
// with and without parse_special.
static const testcase cases[] = {
    {"empty",             "",                                              true,  false},
    {"ascii",             "The capital of France is",                      true,  false},
    {"french",            "L'élève a mangé une crêpe à Nîmes.",            true,  false},
    {"leading_space",     " leading",                                      true,  false},
    {"trailing_space",    "trailing ",                                     true,  false},
    {"double_space",      "two  spaces",                                   true,  false},
    {"only_spaces",       "   ",                                           true,  false},
    {"newline",           "one\ntwo",                                      true,  false},
    {"newline_run",       "a\n\n\n\nb",                                    true,  false},
    {"newline_only",      "\n\n",                                          true,  false},
    {"tab",               "a\tb\t\tc",                                     true,  false},
    {"crlf",              "line\r\nline",                                  true,  false},
    {"digits",            "1234567890 and 42 and 3.14159",                 true,  false},
    {"punctuation",       "Wait... really?! (yes) [no] {maybe} <a>",       true,  false},
    {"emoji",             "bravo 🎉🇫🇷 et voilà",                          true,  false},
    {"cjk",               "日本語のテキストです。",                          true,  false},
    {"mixed_script",      "Привет, мир — hello 世界",                       true,  false},
    {"code",              "func main() {\n\tfmt.Println(\"hi\")\n}",       true,  false},
    {"long_word",         "Donaudampfschiffahrtselektrizitaetenhauptbetriebswerkbauunterbeamtengesellschaft", true, false},
    {"no_bos",            "The capital of France is",                      false, false},
    {"special_literal",   "before <turn|> after",                          true,  false},
    {"special_parsed",    "before <turn|> after",                          true,  true},
    {"special_adjacent",  "<bos><turn|>user\nhello<turn|>",                true,  true},
    {"special_only",      "<eos>",                                         true,  true},
    {"byte_fallback",     "\xef\xbf\xbd unlikely \xe2\x80\x8b zero width",  true,  false},
    {"literal_byte_tok",  "<0x41> is not an A",                            true,  false},
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

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: dump_tokens <model.gguf> <out-dir>\n");
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string out_dir    = argv[2];

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) die("cannot load the model");

    const llama_vocab * vocab = llama_model_get_vocab(model);

    FILE * f = fopen((out_dir + "/cases.json").c_str(), "wb");
    if (!f) die("cannot write cases.json");

    fprintf(f, "{\n  \"cases\": [\n");
    const int n_cases = (int) (sizeof(cases) / sizeof(cases[0]));
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
