#!/usr/bin/env bash
# Copy each CI model's header into the parse corpus, then fuzz Gguf::parse.
set -euo pipefail
cd "$(dirname "$0")"
root="$(cd .. && pwd)"
models="${SCRIBE_MODELS:-$root/models}"

# libFuzzer is C++. When `c++` is clang invoked for the Rust target
# x86_64-unknown-linux-gnu, it misses the host libstdc++ headers, and
# rust-lld does not search the gcc directory that holds libstdc++.so.
if ! echo '#include <cassert>' | c++ -std=c++17 -x c++ -c -o /dev/null - >/dev/null 2>&1; then
  export CXX="${CXX:-g++}"
fi
if command -v g++ >/dev/null; then
  libstd="$(g++ -print-file-name=libstdc++.so)"
  if [[ "$libstd" != libstdc++.so ]]; then
    export LIBRARY_PATH="$(dirname "$libstd")${LIBRARY_PATH:+:$LIBRARY_PATH}"
  fi
fi

mkdir -p corpus/parse
for dest in stories260K.gguf stories15M-q8_0.gguf SmolLM2-135M-Instruct-Q8_0.gguf Qwen3-0.6B-Q8_0.gguf; do
  stem="${dest%.gguf}"
  cargo run -q --manifest-path "$root/Cargo.toml" -p gguf --bin gguf-card -- header "$models/$dest" > "corpus/parse/$stem"
done
cargo +nightly fuzz run parse -- -rss_limit_mb=512 -timeout=10 -max_total_time=600
