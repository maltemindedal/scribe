# 0002: CI checks correctness only

- **Status:** Accepted
- **Source:** [issue #3](https://github.com/maltemindedal/scribe/issues/3),
  comments in [`ci.yml`](../../../.github/workflows/ci.yml) and
  [`bench.yml`](../../../.github/workflows/bench.yml), implemented in
  [PR #47](https://github.com/maltemindedal/scribe/pull/47)

## Context

Scribe needs both correctness checks on every change and, later, performance
numbers. The `bench.yml` comment records the constraint: shared CI runners are
too noisy for performance numbers.

## Decision

- `ci.yml` runs on every pull request and every push to `main`, and checks
  formatting, Clippy and tests. It never measures performance.
- The matrix is `ubuntu-latest` (x86-64) and `macos-latest` (arm64). Issue #3
  and the `ci.yml` comment say the macOS runner must stay because it is the
  first place NEON code actually executes.
- `bench.yml` runs only on manual dispatch. It is a placeholder until the
  benchmark harness ([issue #41](https://github.com/maltemindedal/scribe/issues/41))
  lands.
- Only small models may be added to CI. Issue #3 notes that macOS runners
  have 3 cores and 7 GB RAM, so 1B-class models stay out.

## Consequences

- A green CI run says nothing about speed.
- Tests that need model files stay `#[ignore]` until the model downloader
  ([issue #5](https://github.com/maltemindedal/scribe/issues/5)) is wired into
  the CI cache hook.
