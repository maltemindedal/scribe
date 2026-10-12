#!/usr/bin/env bash
# Drive Scribe the way a contributor or API client does and keep the evidence.
# Usage: verify.sh <doctor|build|test|cli|models|model-test|server|cleanup|all>
set -uo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
RUN_ID="${SCRIBE_VERIFY_RUN:-$(date +%Y%m%d-%H%M%S)}"
ART="${SCRIBE_VERIFY_ARTIFACTS:-/tmp/scribe-verify/$RUN_ID}"
mkdir -p "$ART"
cd "$ROOT" || exit 1

# run <name> <cmd...>: records command, stdout, stderr and exit code under $ART/<name>.*
run() {
  local name="$1"; shift
  printf '%q ' "$@" > "$ART/$name.cmd"; echo >> "$ART/$name.cmd"
  "$@" > "$ART/$name.stdout" 2> "$ART/$name.stderr"
  local rc=$?
  echo "$rc" > "$ART/$name.exit"
  echo "[$name] exit=$rc  ($ART/$name.*)"
  return "$rc"
}

expect() {
  local desc="$1" file="$2" pattern="$3"
  if grep -qE -- "$pattern" "$file"; then echo "  PASS $desc"; else echo "  FAIL $desc (no /$pattern/ in $file)"; FAILED=1; fi
}

FAILED=0
SKIPPED=0

doctor() {
  echo "artifacts: $ART"
  echo "repo: $ROOT @ $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
  echo "arch: $(uname -m)"
  command -v cargo >/dev/null || { echo "  FAIL cargo not on PATH"; return 1; }
  local rv; rv="$(rustc --version)"; echo "rustc: $rv"
  [[ "$rv" == *" 1.99."* || "$rv" == *" 1.1"[0-9][0-9]"."* ]] && echo "  PASS toolchain >= 1.99 (pinned in rust-toolchain.toml)" || { echo "  FAIL toolchain is not 1.99"; return 1; }
  command -v curl >/dev/null && echo "  PASS curl present ($(curl --version | head -1 | cut -d' ' -f2), needs >= 7.71 for models fetch)" || echo "  WARN curl missing: models fetch will fail"
  echo "models dir: ${SCRIBE_MODELS:-$ROOT/models} ($(ls "${SCRIBE_MODELS:-$ROOT/models}" 2>/dev/null | grep -c . ) files)"
}

build() { run build cargo build --workspace; }

test_ws() {
  run test cargo test --workspace || FAILED=1
  expect "every test result line is ok" "$ART/test.stdout" "^test result: ok"
  if grep -E "^test result:" "$ART/test.stdout" | grep -qv "^test result: ok"; then echo "  FAIL a test result line is not ok"; FAILED=1; fi
  grep -E "^test result:" "$ART/test.stdout" > "$ART/test.summary"
  echo "  $(awk '{p+=$4; i+=$8} END {print p" passed, "i" ignored across "NR" binaries"}' "$ART/test.summary")"
}

cli() {
  build || { FAILED=1; return; }
  local bin="$ROOT/target/debug/scribe" ver
  ver="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
  run cli-version "$bin" --version; expect "--version prints 'scribe $ver'" "$ART/cli-version.stdout" "^scribe $ver$"
  run cli-help "$bin" --help; expect "--help prints usage" "$ART/cli-help.stdout" "^Usage: scribe <COMMAND>"
  run cli-noargs "$bin"; expect "no args prints help" "$ART/cli-noargs.stdout" "^Usage: scribe <COMMAND>"
  run cli-unknown "$bin" foo; expect "unknown command exits 1" "$ART/cli-unknown.exit" "^1$"
  expect "unknown command error on stderr" "$ART/cli-unknown.stderr" "^error: unknown command 'foo'"
}

models() {
  run models-fetch cargo xtask models fetch --set "${SCRIBE_VERIFY_SET:-ci}" || FAILED=1
  expect "every pinned model is ok or fetched" "$ART/models-fetch.stdout" "^(ok|fetched) "
  if grep -q "^FAILED" "$ART/models-fetch.stderr"; then echo "  FAIL a model failed verification"; FAILED=1; fi
  ls -l "${SCRIBE_MODELS:-$ROOT/models}" > "$ART/models-ls.txt" 2>&1
}

model_test() {
  run model-test cargo test -p gguf -- --ignored || FAILED=1
  expect "parses_qwen3_0_6b_q8_0 passes" "$ART/model-test.stdout" "^test parses_qwen3_0_6b_q8_0 \.\.\. ok"
}

server() {
  if grep -q "serve" crates/cli/src/main.rs && [ -n "$(sed -n '3,$p' crates/server/src/lib.rs)" ]; then
    echo "  FAIL server code exists now: replace this stub with a real launch/drive (see features/openai-server.md)"; FAILED=1
  else
    echo "  SKIP server: not implemented yet (issue #36). crates/server is an empty crate and 'scribe' has no serve command."
    echo "skipped: #36 not implemented" > "$ART/server.skip"; SKIPPED=1
  fi
}

cleanup() {
  # Nothing long-lived is started today. Kill only PIDs this run recorded; never by name.
  for f in "$ART"/*.pid; do [ -e "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
  echo "cleanup done; evidence kept in $ART"
}

case "${1:-}" in
  doctor) doctor || FAILED=1 ;;
  build) build || FAILED=1 ;;
  test) test_ws ;;
  cli) cli ;;
  models) models ;;
  model-test) model_test ;;
  server) server ;;
  cleanup) cleanup ;;
  all) doctor || FAILED=1; test_ws; cli; server; cleanup ;;
  *) sed -n '2,3p' "$0"; exit 2 ;;
esac
if [ "$FAILED" != 0 ]; then echo "RESULT: FAIL ($1)"; elif [ "$SKIPPED" != 0 ]; then echo "RESULT: PASS with SKIP ($1): skipped features are unverified"; else echo "RESULT: PASS ($1)"; fi
exit "$FAILED"
