## What changed and why

<!-- Describe the change and the reason for it. -->

## Linked issue

Closes #

## Merge method

<!-- merge, squash or rebase, then why. Squash also needs a `Squash message:` line and a fenced block with the commit message. See CONTRIBUTING.md. -->

## Checks

These are the checks CI runs, from [CONTRIBUTING.md](https://github.com/maltemindedal/scribe/blob/main/CONTRIBUTING.md#verification-commands). I ran them before opening this PR:

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] Docs under `docs/` match any behaviour this PR changes.
