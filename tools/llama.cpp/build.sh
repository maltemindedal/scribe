#!/usr/bin/env bash
# Fetch llama.cpp at the commit in PIN and build the CPU-only reference tools:
# scribe-dump, llama-perplexity and llama-tokenize. Idempotent.
# Usage: tools/llama.cpp/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
commit="$(sed -n 's/^commit=//p' "$HERE/PIN")"
tag="$(sed -n 's/^tag=//p' "$HERE/PIN")"
SRC="$HERE/src"
BUILD="$HERE/build"

# CPU only: no GPU backends and no BLAS/Accelerate. On x86-64 GGML_NATIVE is off
# and the ISA is pinned to AVX2/FMA/F16C, so the build does not pick up AVX-512
# or AMX from the build host and any AVX2 machine runs the same kernels. Other
# architectures keep GGML_NATIVE, so their results can differ from x86-64.
# run.sh also passes --no-repack, so Q8_0 weights are never repacked into
# CPU-specific (AMX, or AArch64 interleaved) buffers.
CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
)
case "$(uname -m)" in
  x86_64|amd64)
    CMAKE_FLAGS+=(-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON
      -DGGML_AVX512=OFF -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF) ;;
esac

if [ ! -d "$SRC/.git" ]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
fi
if [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$commit" ]; then
  echo "fetching llama.cpp $tag ($commit)"
  git -C "$SRC" fetch -q --depth 1 origin "$commit"
  git -C "$SRC" checkout -q --detach FETCH_HEAD
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$commit" ] || { echo "error: src is not at $commit" >&2; exit 1; }
if [ -n "${SOURCE_DATE_EPOCH:-}" ]; then
  # llama.cpp turns GGML_NATIVE off when this is set; see research/report.md.
  echo "warning: SOURCE_DATE_EPOCH is set, so GGML_NATIVE is off" >&2
fi

cmake -S "$HERE" -B "$BUILD" "${CMAKE_FLAGS[@]}" > "$BUILD.cmake.log" 2>&1 || { cat "$BUILD.cmake.log" >&2; exit 1; }
cmake --build "$BUILD" -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" \
  --target scribe-dump llama-perplexity llama-tokenize > "$BUILD.build.log" 2>&1 || { tail -40 "$BUILD.build.log" >&2; exit 1; }

BIN="$BUILD/bin"
for b in scribe-dump llama-perplexity llama-tokenize; do
  [ -x "$BIN/$b" ] || { echo "error: $b was not built" >&2; exit 1; }
done
{
  echo "llama_cpp_tag=$tag"
  echo "llama_cpp_commit=$commit"
  echo "cmake_flags=${CMAKE_FLAGS[*]}"
  echo "compiler=$(c++ --version | head -1)"
  echo "host=$(uname -sm)"
  echo "bin=$BIN"
} > "$BUILD/BUILD_INFO"
echo "built llama.cpp $tag into $BIN"
