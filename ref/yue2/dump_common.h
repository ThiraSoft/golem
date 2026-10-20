#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Fixtures {
    std::string dir;
    std::map<std::string, std::vector<int>> shapes;

    explicit Fixtures(std::string d) : dir(std::move(d)) {}

    void write(const std::string & name, const float * data, std::vector<int> shape) {
        fs::path p = fs::path(dir) / name;
        fs::create_directories(p.parent_path());

        size_t n = 1;
        for (int dim : shape) {
            n *= (size_t) dim;
        }

        std::ofstream f(p, std::ios::binary);
        if (!f.is_open()) {
            fprintf(stderr, "Fixtures: cannot write %s\n", p.string().c_str());
            exit(1);
        }
        f.write(reinterpret_cast<const char *>(data), (std::streamsize) (n * sizeof(float)));
        if (!f) {
            fprintf(stderr, "Fixtures: error writing %s\n", p.string().c_str());
            exit(1);
        }
        shapes[name] = std::move(shape);
        fprintf(stderr, "Fixtures: wrote %s (%zu floats)\n", name.c_str(), n);
    }

    void write_i32(const std::string & name, const int32_t * data, std::vector<int> shape) {
        fs::path p = fs::path(dir) / name;
        fs::create_directories(p.parent_path());

        size_t n = 1;
        for (int dim : shape) {
            n *= (size_t) dim;
        }

        std::ofstream f(p, std::ios::binary);
        if (!f.is_open()) {
            fprintf(stderr, "Fixtures: cannot write %s\n", p.string().c_str());
            exit(1);
        }
        f.write(reinterpret_cast<const char *>(data), (std::streamsize) (n * sizeof(int32_t)));
        if (!f) {
            fprintf(stderr, "Fixtures: error writing %s\n", p.string().c_str());
            exit(1);
        }
        shapes[name] = std::move(shape);
        fprintf(stderr, "Fixtures: wrote %s (%zu int32s)\n", name.c_str(), n);
    }

    ~Fixtures() {
        fs::path meta_path = fs::path(dir) / "meta.json";
        std::map<std::string, std::vector<int>> merged;

        if (fs::exists(meta_path)) {
            std::ifstream in(meta_path);
            if (in.is_open()) {
                std::string content((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
                size_t pos = 0;
                while (pos < content.size()) {
                    size_t k_start = content.find('"', pos);
                    if (k_start == std::string::npos) break;
                    size_t k_end = content.find('"', k_start + 1);
                    if (k_end == std::string::npos) break;
                    std::string key = content.substr(k_start + 1, k_end - k_start - 1);

                    size_t colon = content.find(':', k_end);
                    if (colon == std::string::npos) break;

                    size_t bracket_open = content.find('[', colon);
                    if (bracket_open == std::string::npos) break;
                    size_t bracket_close = content.find(']', bracket_open);
                    if (bracket_close == std::string::npos) break;

                    std::string dims_str = content.substr(bracket_open + 1, bracket_close - bracket_open - 1);
                    std::vector<int> dims;
                    std::stringstream ss(dims_str);
                    std::string val;
                    while (std::getline(ss, val, ',')) {
                        size_t start = val.find_first_not_of(" \t\r\n");
                        size_t end = val.find_last_not_of(" \t\r\n");
                        if (start != std::string::npos && end != std::string::npos) {
                            try {
                                dims.push_back(std::stoi(val.substr(start, end - start + 1)));
                            } catch (...) {}
                        }
                    }
                    if (!key.empty() && !dims.empty()) {
                        merged[key] = dims;
                    }
                    pos = bracket_close + 1;
                }
            }
        }

        for (const auto & kv : shapes) {
            merged[kv.first] = kv.second;
        }

        std::map<std::string, std::vector<int>> final_meta;
        for (const auto & kv : merged) {
            if (fs::is_regular_file(fs::path(dir) / kv.first)) {
                final_meta[kv.first] = kv.second;
            }
        }

        fs::create_directories(meta_path.parent_path());
        std::ofstream out(meta_path);
        if (!out.is_open()) {
            fprintf(stderr, "Fixtures: cannot write %s\n", meta_path.string().c_str());
            return;
        }

        out << "{\n";
        bool first = true;
        for (const auto & [key, shape] : final_meta) {
            if (!first) {
                out << ",\n";
            }
            first = false;
            out << " \"" << key << "\": [\n";
            for (size_t i = 0; i < shape.size(); i++) {
                out << "  " << shape[i];
                if (i + 1 < shape.size()) {
                    out << ",";
                }
                out << "\n";
            }
            out << " ]";
        }
        out << "\n}\n";
        fprintf(stderr, "Fixtures: updated %s (%zu entries)\n", meta_path.string().c_str(), final_meta.size());
    }
};
