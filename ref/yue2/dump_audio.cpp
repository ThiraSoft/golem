// dump_audio: turns the outputs of yue2.cpp's own CLI into fixtures, and
// records its output stage.
//
//   dump_audio <fixture dir> <name> <song.wav> <song.vae> [normalize]
//
// yue-synth with output_format wav32 writes the decoder's float audio with
// no normalization; this reads it back to planar [2][T] under
// pipeline/<name>/audio, and the .vae it wrote with --latent, time-major
// [T_lat][64], under pipeline/<name>/latent. With "normalize", the same audio
// also goes through audio_normalize at three peak_clip values and through
// the wav16, wav24 and wav32 encoders (normalize/*), which is what golem's
// Song.Encode is held to byte for byte.

#include "audio-io.h"
#include "dump_common.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static std::vector<float> read_floats(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "dump_audio: cannot read %s\n", path);
        exit(1);
    }
    size_t bytes = (size_t) f.tellg();
    f.seekg(0);
    std::vector<float> out(bytes / sizeof(float));
    f.read(reinterpret_cast<char *>(out.data()), (std::streamsize) (out.size() * sizeof(float)));
    return out;
}

// Raw bytes under name, their count as the shape.
static void write_bytes(Fixtures & fx, const std::string & name, const std::string & bytes) {
    fs::path p = fs::path(fx.dir) / name;
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f.write(bytes.data(), (std::streamsize) bytes.size());
    if (!f) {
        fprintf(stderr, "dump_audio: cannot write %s\n", p.string().c_str());
        exit(1);
    }
    fx.shapes[name] = { (int) bytes.size() };
    fprintf(stderr, "Fixtures: wrote %s (%zu bytes)\n", name.c_str(), bytes.size());
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <fixture dir> <name> <song.wav> <song.vae> [normalize]\n", argv[0]);
        return 1;
    }
    Fixtures    fx(argv[1]);
    std::string name = argv[2];

    int     T = 0, sr = 0;
    float * planar = audio_read(argv[3], &T, &sr);
    if (!planar || sr != 48000) {
        fprintf(stderr, "dump_audio: %s is not 48 kHz audio\n", argv[3]);
        return 1;
    }
    std::vector<float> audio(planar, planar + (size_t) 2 * T);
    free(planar);
    fx.write("pipeline/" + name + "/audio", audio.data(), { 2, T });

    std::vector<float> latent = read_floats(argv[4]);
    fx.write("pipeline/" + name + "/latent", latent.data(), { (int) (latent.size() / 64), 64 });

    if (argc > 5 && std::string(argv[5]) == "normalize") {
        for (int clip : { 0, 10, 999 }) {
            std::vector<float> v = audio;
            audio_normalize(v.data(), T * 2, clip);
            fx.write("normalize/out_" + std::to_string(clip), v.data(), { 2, T });
            if (clip == 10) {
                write_bytes(fx, "normalize/wav16", audio_encode_wav(v.data(), T, 48000, WAV_S16));
                write_bytes(fx, "normalize/wav24", audio_encode_wav(v.data(), T, 48000, WAV_S24));
            }
        }
        write_bytes(fx, "normalize/wav32", audio_encode_wav(audio.data(), T, 48000, WAV_F32));
    }
    return 0;
}
