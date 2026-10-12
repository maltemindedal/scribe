# Scribe

Scribe is a from-scratch LLM inference engine in Rust. It targets
Qwen3-0.6B Q8_0 on CPU (Apple silicon first, x86-64 too), is debugged on tiny
models, verifies every layer against Hugging Face float32 and llama.cpp
references, and serves an OpenAI-compatible API.

The build order and milestones live in the roadmap issue:
[#1](https://github.com/maltemindedal/scribe/issues/1).

## Layout

| Crate | Owns |
|---|---|
| `gguf` | GGUF file parsing, mmap, tensor views |
| `tokenizer` | BPE (later SentencePiece), chat templates |
| `kernels` | Block types, scalar and SIMD kernels, thread pool |
| `model` | Config, weights, forward pass, KV cache |
| `sampler` | Logit processing and token sampling |
| `engine` | Slots, prefix cache, scheduler (no HTTP, no JSON) |
| `server` | HTTP and the OpenAI-compatible API |
| `bench` | Benchmarks |
| `cli` | The `scribe` binary |

Only `kernels` (and the mmap module in `gguf`) may use `unsafe`; every other
crate forbids it.

Model files are never committed; `models/` and `*.gguf` are gitignored.

## Building

```sh
cargo build --workspace
cargo test --workspace
cargo run -p cli -- --help
```

The toolchain is pinned in `rust-toolchain.toml` (Rust 1.99 or newer).

## Licence

MIT, see [LICENSE](LICENSE).
