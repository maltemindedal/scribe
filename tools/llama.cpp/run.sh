#!/usr/bin/env bash
# Build llama.cpp (build.sh) and write reference dumps for Scribe (#21):
# activations and all-position logits for fixed token ids, plus
# llama-perplexity's own KL-divergence numbers on text.txt.
#
# Usage: tools/llama.cpp/run.sh [model-stem ...]
#   default models: stories15M-q8_0 Qwen3-0.6B-Q8_0 (fetch them with `cargo xtask models fetch`)
#   LENGTHS="1 17 128" sets the sequence lengths (token ids from tools/llama.cpp/tokens/).
# Output: $SCRIBE_MODELS/reference/llama.cpp/<model>/<len>/ (default models/), each with
# index.tsv, *.f32, tokens.txt, BUILD_INFO and SHA256SUMS; <model>/kld/ holds the
# llama-perplexity logs.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODELS="${SCRIBE_MODELS:-$ROOT/models}"
OUT="$MODELS/reference/llama.cpp"
LENGTHS="${LENGTHS:-1 17 128}"
THREADS="${THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
# --no-repack keeps Q8_0 weights in plain CPU buffers. Without it the CPU
# backend repacks them for AMX (or AArch64) kernels, which made two runs of the
# same build disagree (Qwen3 128 tokens, 8 threads, native AMX build: mean KLD
# 5.6e-3, 3.9e-3 to 9.9e-3 across runs; see README.md) and ties results to the CPU.
LLAMA_ARGS=(-t "$THREADS" -nr)
[ "$#" -gt 0 ] || set -- stories15M-q8_0 Qwen3-0.6B-Q8_0

bash "$HERE/build.sh"
BIN="$(sed -n 's/^bin=//p' "$HERE/build/BUILD_INFO")"

for model in "$@"; do
  gguf="$MODELS/$model.gguf"
  [ -f "$gguf" ] || { echo "error: $gguf not found; run cargo xtask models fetch" >&2; exit 1; }
  for len in $LENGTHS; do
    ids="$HERE/tokens/$model-$len.txt"
    [ -f "$ids" ] || { echo "error: no token ids $ids" >&2; exit 1; }
    dir="$OUT/$model/$len"
    rm -rf "$dir" "$dir.log"; mkdir -p "$(dirname "$dir")"  # scribe-dump creates $dir
    "$BIN/scribe-dump" --tokens "$ids" --out "$dir" -m "$gguf" "${LLAMA_ARGS[@]}" > "$dir.log" 2>&1 \
      || { tail -20 "$dir.log" >&2; exit 1; }
    mv "$dir.log" "$dir/scribe-dump.log"
    cp "$ids" "$dir/tokens.txt"
    { cat "$HERE/build/BUILD_INFO"; echo "model=$model.gguf"; echo "model_sha256=$(sha256sum "$gguf" | cut -d' ' -f1)"; echo "threads=$THREADS"; echo "llama_args=${LLAMA_ARGS[*]}"; echo "$(grep -m1 "system_info" "$dir/scribe-dump.log")"; } > "$dir/BUILD_INFO"
    (cd "$dir" && sha256sum -- *.f32 tokens.txt index.tsv > SHA256SUMS)
    echo "dumped $model len=$len: $(($(wc -l < "$dir/index.tsv") - 1)) tensors + logits -> $dir"
  done

  # llama.cpp's own KLD: the model against a base written by the same model, so the
  # expected mean KLD is 0. Comparing another build or quantisation uses the same base.
  kdir="$OUT/$model/kld"
  rm -rf "$kdir"; mkdir -p "$kdir"
  "$BIN/llama-perplexity" -m "$gguf" -f "$HERE/text.txt" -c 128 -b 128 "${LLAMA_ARGS[@]}" \
    --kl-divergence-base "$kdir/base.kld" > "$kdir/base.log" 2>&1 || { tail -20 "$kdir/base.log" >&2; exit 1; }
  "$BIN/llama-perplexity" -m "$gguf" -c 128 -b 128 "${LLAMA_ARGS[@]}" \
    --kl-divergence-base "$kdir/base.kld" --kl-divergence > "$kdir/self.log" 2>&1 || { tail -20 "$kdir/self.log" >&2; exit 1; }
  { cat "$HERE/build/BUILD_INFO"; echo "threads=$THREADS"; echo "llama_args=${LLAMA_ARGS[*]}"; grep -m1 "system_info" "$kdir/base.log"; } > "$kdir/BUILD_INFO"
  echo "llama-perplexity self-KLD for $model: $(grep -E '^Mean +KLD' "$kdir/self.log" | head -1)"
done
