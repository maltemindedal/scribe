// scribe-dump: run one teacher-forced pass of a GGUF model through llama.cpp
// and write full f32 tensors for Scribe to compare against (#21).
//
// Usage:
//   scribe-dump --tokens ids.txt --out DIR [--filter REGEX] -m model.gguf [llama.cpp args]
//
// ids.txt holds whitespace-separated token ids. Every position is an output,
// so DIR/logits.f32 has n_vocab x n_tokens values. Tensors whose name matches
// REGEX (default: the names the Scribe comparator knows) are written as
// DIR/<name>.f32, in ggml element order (ne0 fastest), converted to f32.
// DIR/index.tsv lists one tensor per line: name, ggml type, ne0,ne1,ne2,ne3, file.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char * DEFAULT_FILTER =
    "^(attn_norm|Qcur|Kcur|Vcur|Qcur_normed|Kcur_normed|kqv_out|attn_out|ffn_inp|ffn_norm|"
    "ffn_gate|ffn_up|ffn_swiglu|ffn_out|l_out)-[0-9]+$|^(result_norm|result_output)$";

struct dump_state {
    std::string out_dir;
    std::regex filter;
    // name -> index line; a name seen again overwrites the earlier tensor.
    std::map<std::string, std::string> written;
    std::vector<uint8_t> buf;
    int overwritten = 0;
};

float value_at(const uint8_t * data, ggml_type type, size_t off) {
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) (data + off);
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) (data + off));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) (data + off));
        default: throw std::runtime_error("unsupported tensor type");
    }
}

bool write_f32(const std::string & path, const std::vector<float> & v) {
    std::FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const bool ok = std::fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
    return std::fclose(f) == 0 && ok;
}

bool on_tensor(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const bool float_type = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
    const bool wanted = float_type && std::regex_search(t->name, st->filter);
    if (ask) {
        return wanted;
    }
    if (!wanted) {
        return true;
    }

    const size_t nbytes = ggml_nbytes(t);
    const uint8_t * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *) t->data;
    } else {
        st->buf.resize(nbytes);
        ggml_backend_tensor_get(t, st->buf.data(), 0, nbytes);
        data = st->buf.data();
    }

    std::vector<float> out;
    out.reserve(ggml_nelements(t));
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
        out.push_back(value_at(data, t->type, i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]));
    }

    const std::string file = std::string(t->name) + ".f32";
    if (!write_f32(st->out_dir + "/" + file, out)) {
        std::fprintf(stderr, "scribe-dump: failed to write %s\n", file.c_str());
        return false;
    }
    // llama.cpp gives some names to two nodes (attn_norm-N names the normalised
    // input and then its product with the norm weight); the last one is the
    // tensor the name stands for, so a later sighting replaces the earlier.
    char line[512];
    std::snprintf(line, sizeof line, "%s\t%s\t%lld,%lld,%lld,%lld\t%s\n", t->name, ggml_type_name(t->type),
                  (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], file.c_str());
    st->overwritten += st->written.count(t->name);
    st->written[t->name] = line;
    return true;
}

std::vector<llama_token> read_ids(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<llama_token> ids;
    long long id;
    while (in >> id) {
        ids.push_back((llama_token) id);
    }
    if (!in.eof()) {
        throw std::runtime_error(path + ": not a list of integer token ids");
    }
    return ids;
}

}  // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string tokens_path, out_dir, filter = DEFAULT_FILTER;
    std::vector<char *> rest = {argv[0]};
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if ((a == "--tokens" || a == "--out" || a == "--filter") && i + 1 < argc) {
            (a == "--tokens" ? tokens_path : a == "--out" ? out_dir : filter) = argv[++i];
        } else {
            rest.push_back(argv[i]);
        }
    }
    if (tokens_path.empty() || out_dir.empty()) {
        std::fprintf(stderr, "usage: %s --tokens ids.txt --out DIR [--filter REGEX] -m model.gguf [llama.cpp args]\n", argv[0]);
        return 2;
    }

    std::vector<llama_token> ids;
    try {
        ids = read_ids(tokens_path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "scribe-dump: %s\n", e.what());
        return 1;
    }
    if (ids.empty()) {
        std::fprintf(stderr, "scribe-dump: %s has no token ids\n", tokens_path.c_str());
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    // One ubatch, so every tensor is seen exactly once and covers all positions.
    const int n = (int) ids.size();
    params.n_ctx = std::max(params.n_ctx, n);
    params.n_batch = std::max(params.n_batch, n);
    params.n_ubatch = std::max(params.n_ubatch, n);
    params.warmup = false;

    dump_state st;
    st.out_dir = out_dir;
    st.filter = std::regex(filter, std::regex::optimize);
    params.cb_eval = on_tensor;
    params.cb_eval_user_data = &st;

    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) {
        std::fprintf(stderr, "scribe-dump: failed to load the model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token id : ids) {
        if (id < 0 || id >= n_vocab) {
            std::fprintf(stderr, "scribe-dump: token id %d is outside the vocabulary (%d)\n", id, n_vocab);
            return 1;
        }
    }

    common_batch batch(ctx);
    for (int i = 0; i < n; i++) {
        batch.add(ids[i], i, 0, true);
    }
    if (llama_n_ubatch(ctx) < (uint32_t) n) {
        std::fprintf(stderr, "scribe-dump: n_ubatch %u < %d tokens; the pass would be split\n", llama_n_ubatch(ctx), n);
        return 1;
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        std::fprintf(stderr, "scribe-dump: decode failed\n");
        return 1;
    }

    std::vector<float> logits;
    logits.reserve((size_t) n_vocab * n);
    for (int i = 0; i < n; i++) {
        const float * row = llama_get_logits_ith(ctx, i);
        if (!row) {
            std::fprintf(stderr, "scribe-dump: no logits for position %d\n", i);
            return 1;
        }
        logits.insert(logits.end(), row, row + n_vocab);
    }
    if (!write_f32(out_dir + "/logits.f32", logits)) {
        std::fprintf(stderr, "scribe-dump: failed to write logits.f32\n");
        return 1;
    }
    std::FILE * index = std::fopen((out_dir + "/index.tsv").c_str(), "w");
    if (!index) {
        std::fprintf(stderr, "scribe-dump: cannot write %s/index.tsv\n", out_dir.c_str());
        return 1;
    }
    for (const auto & kv : st.written) {
        std::fputs(kv.second.c_str(), index);
    }
    std::fprintf(index, "logits\tf32\t%d,%d,1,1\tlogits.f32\n", n_vocab, n);
    if (std::fclose(index) != 0) {
        std::fprintf(stderr, "scribe-dump: failed to write index.tsv\n");
        return 1;
    }
    std::printf("scribe-dump: %zu tensors (%d same-name nodes replaced) and logits [%d x %d] in %s\n",
                st.written.size(), st.overwritten, n_vocab, n, out_dir.c_str());
    llama_backend_free();
    return 0;
}
