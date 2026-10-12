# llama.cpp reference dumps

Rung 5 of the verification ladder (#21) compares Scribe against llama.cpp on
the same GGUF file. This directory builds llama.cpp at a pinned commit, CPU
only. It writes full activation and logit dumps for fixed token ids, and it
records llama.cpp's own KL-divergence numbers. The `bench` crate reads the
dumps and compares them.

## One command

```sh
cargo xtask models fetch          # stories15M-q8_0 and Qwen3-0.6B-Q8_0, among others
tools/llama.cpp/run.sh
```

`run.sh` needs `git`, `cmake` and a C++17 compiler. On the first run it fetches
llama.cpp at the commit in [`PIN`](PIN) (`b11576`) into `src/` and builds it
into `build/`; both are gitignored. It then writes, for each model and for
`LENGTHS` (default `1 17 128`):

```text
models/reference/llama.cpp/<model>/<len>/
  index.tsv       one line per tensor: name, ggml type, ne0,ne1,ne2,ne3, file
  <name>.f32      raw little-endian f32, ggml order (ne0 fastest)
  logits.f32      all positions: ne0 = n_vocab, ne1 = n_tokens
  tokens.txt      the token ids that were run
  BUILD_INFO      llama.cpp tag, commit, cmake flags, compiler, host, model SHA-256, threads
  SHA256SUMS
models/reference/llama.cpp/<model>/kld/
  base.kld, base.log, self.log, BUILD_INFO   llama-perplexity --kl-divergence-base / --kl-divergence on text.txt
```

`SCRIBE_MODELS` moves the whole tree, as it does for the model tests. Pass
model stems to dump only some models, for example `tools/llama.cpp/run.sh stories15M-q8_0`.
`THREADS` sets the thread count, which defaults to 1 (see below).

On an 8-core x86-64 box, a run from an empty `src/` and `build/` took 105 s,
including the fetch and the build. With the build already in place, it took 16 s. The dumps take about
590 MB, most of it the 128-token Qwen3 dump.

## What is dumped

`scribe-dump` ([`scribe_dump.cpp`](scribe_dump.cpp)) loads the model with
llama.cpp and decodes the token ids as one teacher-forced batch, with every
position marked as an output. It writes every tensor whose name matches
`--filter`. By default those are the llama.cpp graph names `attn_norm`,
`Qcur`, `Kcur`, `Vcur`, `Qcur_normed`, `Kcur_normed`, `kqv_out`, `attn_out`,
`ffn_inp`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_swiglu`, `ffn_out` and
`l_out`, each with `-<layer>`, plus `result_norm` and `result_output`. Names
that a model's graph does not produce are absent: Qwen3 has 394 tensors and
stories15M 80. llama.cpp gives some names to two nodes; `attn_norm-N`, for
example, names the normalised input and then its product with the weight.
The last node with a name is the one kept. The tool refuses to run if the
batch would be split, so every tensor covers all positions.

The token ids in [`tokens/`](tokens) are the first 1, 17 and 128 ids that each
model's tokenizer gives for [`text.txt`](text.txt), from
`build/bin/llama-tokenize -m <model> -f text.txt --ids`. They are committed so
that Scribe runs exactly the same ids.

## Comparing

```sh
cargo run --release -p bench --bin compare -- REF_DIR TEST_DIR [--max-kld X] [--min-same-top PCT]
cargo run --release -p bench --bin compare -- REF_DIR - --perturb 1e-3 --max-kld 0   # must FAIL
```

`compare` prints the max absolute difference, the NMSE and the `check-nmse.py`
band for every tensor. For `logits` it prints mean and max KLD (`KL(P_ref ||
P_test)` per position, as `llama-perplexity` defines it) and the same-top-token
percentage. It exits 1 on a tensor in the bad band (NMSE >= 1e-2) or a broken
bound. `--perturb REL` compares the reference with a deterministically
perturbed copy of itself. That is the self-test that shows the check can fail.

Measured on b11576, linux x86-64, 1 thread, 17 tokens:

| Check | stories15M-q8_0 | Qwen3-0.6B-Q8_0 |
|---|---|---|
| `compare` dump vs itself | mean KLD 0, same top 100% | mean KLD 0, same top 100% |
| `compare --perturb 1e-3` | mean KLD 2.3e-5, NMSE 1.0e-6 | mean KLD 8.3e-5, NMSE 1.0e-6 |
| `llama-perplexity` self KLD on `text.txt` (`kld/self.log`) | 0.000000 ± 0.000001 (both runs) | 0.000111 ± 0.000023 on one run, -0.000001 ± 0.000001 on a fresh rebuild |

For scale, llama.cpp's README reports BF16 vs FP16 at mean KLD 2.5e-5, and
Q8_0 vs f16 at about 1.4e-3 with 97.7% same top token.

## Gotchas

- **Threads change the numbers.** With 8 threads, two runs of the same build
  on the same Qwen3 input gave different tensors from layer 1 onwards. The
  logits differed by mean KLD about 2e-3 and same top 94%. With 1 thread,
  two runs were bit-identical. That is why `THREADS` defaults to 1. A
  comparison against a multi-threaded dump has a noise floor about as large
  as the Q8_0-vs-f16 gap.
- `llama-perplexity`'s self KLD for Qwen3 was 1.1e-4 on one 1-thread run
  and about 0 on another, after a rebuild from scratch. The cause is not
  verified. The exact zero check is `compare` on the f32 dumps.
- `--kl-divergence` needs at least `2 * n_ctx` tokens, so `run.sh` uses
  `-c 128` on the roughly 500-token `text.txt`.
- If `SOURCE_DATE_EPOCH` is set, llama.cpp turns `GGML_NATIVE` off.
  `build.sh` warns about it.
- The dump format is plain TSV, not JSON, because the workspace has no
  third-party dependencies to parse JSON with. #20's Hugging Face dumps
  should write the same `index.tsv` so that the same comparator reads both.
