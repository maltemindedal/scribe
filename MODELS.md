# Test models

Model files never go in the repository. `cargo xtask models fetch` downloads
them into `models/` (gitignored) from pinned Hugging Face revisions and
verifies size and SHA-256, so tests and CI are reproducible.

```sh
cargo xtask models fetch              # the ci set (default)
cargo xtask models fetch --set all    # ci plus the large reference models
```

The pins live in [`crates/xtask/models.toml`](crates/xtask/models.toml): per
model, the Hugging Face repo, file, commit revision, SHA-256, size, licence
and role. CI caches `models/` keyed on that file's hash and fetches the `ci`
set before running the ignored model tests. See the
[how-to guide](docs/guides/running-the-model-tests.md).

## Models

| Model | Set | Source | Licence | Size | Role and what it exercises |
|---|---|---|---|---|---|
| stories260K (F32, SPM vocab) | ci | [`ggml-org/tiny-llamas`](https://huggingface.co/ggml-org/tiny-llamas) | Not declared (see below) | 1,185,376 B | GQA head-mapping, context overflow. |
| stories15M-q8_0 | ci | [`ggml-org/tiny-llamas`](https://huggingface.co/ggml-org/tiny-llamas) | Not declared (see below) | 26,671,328 B | First quantised file. |
| SmolLM2-135M-Instruct Q8_0 | ci | [`bartowski/SmolLM2-135M-Instruct-GGUF`](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF) | Apache-2.0 | 144,811,360 B | First chat model. |
| Qwen3-0.6B Q8_0 | ci | [`Qwen/Qwen3-0.6B-GGUF`](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) | Apache-2.0 | 639,446,688 B | Primary acceptance model; used by the parser's ignored test `parses_qwen3_0_6b_q8_0`. |
| Qwen3-0.6B safetensors (bf16) | all | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache-2.0 | 1,503,300,328 B | Hugging Face float32 reference only ([#20](https://github.com/maltemindedal/scribe/issues/20)). Saved as `Qwen3-0.6B-bf16.safetensors`. |
| Qwen2.5-0.5B-Instruct Q4_K_M | all | [`Qwen/Qwen2.5-0.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | Apache-2.0 | 491,400,032 B | Mixed-quant file for [#42](https://github.com/maltemindedal/scribe/issues/42). |

_As of 2026-10; source of truth: `crates/xtask/models.toml`, whose values come
from the Hugging Face API (`/api/models/<repo>?blobs=true`)._

### tiny-llamas licence

`ggml-org/tiny-llamas` declares no licence: its model card is the single line
"Purely for testing and fun ;)" and the API returns no `license` tag. The
files are used here only as local test fixtures and are never redistributed
from this repository.

## Changing a pin

1. Get the new commit and LFS hash from
   `https://huggingface.co/api/models/<repo>?blobs=true` (`sha` and each
   file's `lfs.sha256` and `size`).
2. Update `revision`, `sha256` and `size` in `crates/xtask/models.toml`. This
   also changes the CI cache key.
3. Delete the old file from `models/` and run `cargo xtask models fetch`.
