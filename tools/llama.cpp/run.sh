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
# One thread by default: with 8 threads two runs of this build on the same
# input gave different Qwen3 logits (mean KLD about 2e-3); with 1 thread they were
# bit-identical. See README.md.
THREADS="${THREADS:-1}"
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
    rm -rf "$dir"; mkdir -p "$dir"
    "$BIN/scribe-dump" --tokens "$ids" --out "$dir" -m "$gguf" -t "$THREADS" > "$dir/scribe-dump.log" 2>&1 \
      || { tail -20 "$dir/scribe-dump.log" >&2; exit 1; }
    cp "$ids" "$dir/tokens.txt"
    { cat "$HERE/build/BUILD_INFO"; echo "model=$model.gguf"; echo "model_sha256=$(sha256sum "$gguf" | cut -d' ' -f1)"; echo "threads=$THREADS"; } > "$dir/BUILD_INFO"
    (cd "$dir" && sha256sum -- *.f32 tokens.txt index.tsv > SHA256SUMS)
    echo "dumped $model len=$len: $(($(wc -l < "$dir/index.tsv") - 1)) tensors + logits -> $dir"
  done

  # llama.cpp's own KLD: the model against a base written by the same model, so the
  # expected mean KLD is 0. Comparing another build or quantisation uses the same base.
  kdir="$OUT/$model/kld"
  rm -rf "$kdir"; mkdir -p "$kdir"
  "$BIN/llama-perplexity" -m "$gguf" -f "$HERE/text.txt" -c 128 -b 128 -t "$THREADS" \
    --kl-divergence-base "$kdir/base.kld" > "$kdir/base.log" 2>&1 || { tail -20 "$kdir/base.log" >&2; exit 1; }
  "$BIN/llama-perplexity" -m "$gguf" -c 128 -b 128 -t "$THREADS" \
    --kl-divergence-base "$kdir/base.kld" --kl-divergence > "$kdir/self.log" 2>&1 || { tail -20 "$kdir/self.log" >&2; exit 1; }
  cp "$HERE/build/BUILD_INFO" "$kdir/BUILD_INFO"
  echo "llama-perplexity self-KLD for $model: $(grep -E '^Mean +KLD' "$kdir/self.log" | head -1)"
done
